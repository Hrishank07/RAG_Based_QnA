"""CDK stack that provisions the Codex retrieval infrastructure."""

from __future__ import annotations

import json
from typing import Any, Dict

from aws_cdk import (  # type: ignore[import]
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_dynamodb as dynamodb  # type: ignore[import]
from aws_cdk import aws_iam as iam  # type: ignore[import]
from aws_cdk import aws_lambda as _lambda  # type: ignore[import]
from aws_cdk import aws_lambda_event_sources as lambda_events  # type: ignore[import]
from aws_cdk import aws_logs as logs  # type: ignore[import]
from aws_cdk import aws_opensearchserverless as aoss  # type: ignore[import]
from aws_cdk import aws_s3 as s3  # type: ignore[import]
from constructs import Construct  # type: ignore[import]


class CodexStack(Stack):
    """Stack containing the storage, indexing, and compute resources."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(
            self,
            "CodexDocs",
            bucket_name=None,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )

        documents_table = dynamodb.Table(
            self,
            "DocumentsTable",
            partition_key=dynamodb.Attribute(name="doc_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="version", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery=True,
            table_name=None,
        )

        chunks_table = dynamodb.Table(
            self,
            "ChunksTable",
            partition_key=dynamodb.Attribute(name="doc_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="chunk_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery=True,
            table_name=None,
        )

        encryption_policy = aoss.CfnSecurityPolicy(
            self,
            "OpenSearchEncryptionPolicy",
            name="codex-os-encryption",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "Resource": [
                                "collection/codex-search",
                                "collection/codex-vector",
                            ],
                            "ResourceType": "collection",
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )

        network_policy = aoss.CfnSecurityPolicy(
            self,
            "OpenSearchNetworkPolicy",
            name="codex-os-network",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": [
                                    "collection/codex-search",
                                    "collection/codex-vector",
                                ],
                                "ResourceType": "collection",
                            }
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )
        network_policy.add_dependency(encryption_policy)

        search_collection = aoss.CfnCollection(
            self,
            "SearchCollection",
            name="codex-search",
            description="Full-text search collection for Codex",
            type="SEARCH",
        )
        search_collection.add_dependency(network_policy)

        vector_collection = aoss.CfnCollection(
            self,
            "VectorCollection",
            name="codex-vector",
            description="Vector search collection for Codex",
            type="VECTORSEARCH",
        )
        vector_collection.add_dependency(network_policy)

        ingest_role = iam.Role(
            self,
            "IngestLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        bucket.grant_read_write(ingest_role)
        documents_table.grant_read_write_data(ingest_role)
        chunks_table.grant_read_write_data(ingest_role)

        ingest_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "textract:StartDocumentTextDetection",
                    "textract:GetDocumentTextDetection",
                    "textract:DetectDocumentText",
                ],
                resources=["*"],
            )
        )

        ingest_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelEndpoint",
                ],
                resources=["*"],
            )
        )

        ingest_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "aoss:APIAccessAll",  # simplified for bootstrap; tighten later
                ],
                resources=["*"],
            )
        )

        access_policy = aoss.CfnAccessPolicy(
            self,
            "OpenSearchAccessPolicy",
            name="codex-os-access",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": [
                                    "collection/codex-search",
                                    "collection/codex-vector",
                                ],
                                "Permission": [
                                    "aoss:DescribeCollectionItems",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                            }
                        ],
                        "Principal": [ingest_role.role_arn],
                    }
                ]
            ),
        )
        access_policy.add_dependency(search_collection)
        access_policy.add_dependency(vector_collection)

        search_index_name = "codex-chunks-bm25"
        vector_index_name = "codex-chunks-vector"

        ingest_function = _lambda.Function(
            self,
            "IngestFunction",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset("codex/lambdas/ingest"),
            environment=self._lambda_env(
                bucket=bucket.bucket_name,
                documents_table=documents_table.table_name,
                chunks_table=chunks_table.table_name,
                search_collection=search_collection.attr_id,
                search_collection_endpoint=search_collection.attr_collection_endpoint,
                vector_collection=vector_collection.attr_id,
                vector_collection_endpoint=vector_collection.attr_collection_endpoint,
                search_index_name=search_index_name,
                vector_index_name=vector_index_name,
            ),
            timeout=Duration.minutes(5),
            log_retention=logs.RetentionDays.ONE_MONTH,
            role=ingest_role,
        )

        ingest_function.add_event_source(
            lambda_events.S3EventSource(
                bucket,
                events=[s3.EventType.OBJECT_CREATED],
                filters=[s3.NotificationKeyFilter(prefix="uploads/")],
            )
        )

        query_role = iam.Role(
            self,
            "QueryLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )
        documents_table.grant_read_data(query_role)
        chunks_table.grant_read_data(query_role)

        query_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelEndpoint"],
                resources=["*"],
            )
        )

        query_role.add_to_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=["*"],
            )
        )

        query_function = _lambda.Function(
            self,
            "QueryFunction",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset("codex/lambdas/query"),
            environment=self._lambda_env(
                bucket=bucket.bucket_name,
                documents_table=documents_table.table_name,
                chunks_table=chunks_table.table_name,
                search_collection=search_collection.attr_id,
                search_collection_endpoint=search_collection.attr_collection_endpoint,
                vector_collection=vector_collection.attr_id,
                vector_collection_endpoint=vector_collection.attr_collection_endpoint,
                search_index_name=search_index_name,
                vector_index_name=vector_index_name,
            ),
            timeout=Duration.seconds(30),
            log_retention=logs.RetentionDays.ONE_MONTH,
            role=query_role,
        )

        self.outputs = {
            "DocsBucket": bucket.bucket_name,
            "DocumentsTable": documents_table.table_name,
            "ChunksTable": chunks_table.table_name,
            "SearchCollectionId": search_collection.attr_id,
            "VectorCollectionId": vector_collection.attr_id,
            "IngestLambdaArn": ingest_function.function_arn,
            "QueryLambdaArn": query_function.function_arn,
        }

    @staticmethod
    def _lambda_env(
        *,
        bucket: str,
        documents_table: str,
        chunks_table: str,
        search_collection: str,
        search_collection_endpoint: str,
        vector_collection: str,
        vector_collection_endpoint: str,
        search_index_name: str,
        vector_index_name: str,
    ) -> Dict[str, str]:
        return {
            "DOCS_BUCKET": bucket,
            "DOCUMENTS_TABLE": documents_table,
            "CHUNKS_TABLE": chunks_table,
            "SEARCH_COLLECTION_ID": search_collection,
            "SEARCH_COLLECTION_ENDPOINT": search_collection_endpoint,
            "VECTOR_COLLECTION_ID": vector_collection,
            "VECTOR_COLLECTION_ENDPOINT": vector_collection_endpoint,
            "SEARCH_INDEX_NAME": search_index_name,
            "VECTOR_INDEX_NAME": vector_index_name,
        }
