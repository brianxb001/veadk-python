# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

from typing import Any

from pydantic import Field
from typing_extensions import override

from veadk.auth.veauth.viking_mem0_veauth import get_viking_mem0_token
from veadk.configs.database_configs import AWSMEMConfig
from veadk.memory.long_term_memory_backends.base_backend import (
    BaseLongTermMemoryBackend,
)
from veadk.utils.logger import get_logger
logger = logging.getLogger(__name__)


class AWSMEMBackend(BaseLongTermMemoryBackend):
    """Mem0 long term memory backend implementation"""

    aws_config: AWSMEMConfig = Field(default_factory=AWSMEMConfig)
    region_name: str = Field(default="us-east-1")
    def model_post_init(self, __context: Any) -> None:
        """Initialize Mem0 client"""
        session = boto3.Session()
        self._data_plane_client = session.client(
            "bedrock-agentcore", region_name=self.aws_config.region_name,
        )
        
        env_memory_type = os.getenv("DATABASE_AWSMEM_MEMORY_TYPE")
        if env_memory_type:
            # "event_1, event_2" -> ["event_1", "event_2"]
            self.memory_type = [x.strip() for x in env_memory_type.split(",")]

    def precheck_index_naming(self):
        """Check if the index name is valid
        For Mem0, there are no specific naming constraints
        """
        pass

    @override
    def save_memory(
        self, event_strings: list[str], user_id: str = "default_user", **kwargs
    ) -> bool:
        """Save memory to Mem0

        Args:
            event_strings: List of event strings to save
            **kwargs: Additional parameters, including 'user_id' for Mem0

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            logger.info(
                f"Saving {len(event_strings)} events to Mem0 for user: {user_id}"
            )

            for event_string in event_strings:
                # Save event string to Mem0
                result = self._mem0_client.add(
                    [{"role": "user", "content": event_string}],
                    user_id=user_id,
                    output_format="v1.1",
                    async_mode=True,
                )
                logger.debug(f"Saved memory result: {result}")

            logger.info(f"Successfully saved {len(event_strings)} events to Mem0")
            return True
        except Exception as e:
            logger.error(f"Failed to save memory to Mem0: {str(e)}")
            return False

    @override
    def search_memory(
        self, query: str, top_k: int, user_id: str = "default_user", **kwargs
    ) -> list[str]:
        """Search memory from Mem0

        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional parameters, including 'user_id' for Mem0

        Returns:
            list[str]: List of memory strings
        """
        memory_records = {
            "preference": [],
            "semantic": [],
            "summary": []
        }
        for memory_type in self.memory_type:
            namespace_prefix = f"/strategies/{memory_type}/actors/{user_id}/sessions/longmemeval"
            
            logger.info("  -> Querying long-term memory in namespace '%s' with query: '%s'...", namespace_prefix, query)
            search_criteria = {"searchQuery": query, "topK": top_k}

            namespace = namespace_prefix
            params = {
                "memoryId": self.aws_config.memory_id,
                "searchCriteria": search_criteria,
                "namespace": namespace,
            }

            try:
                response = self._data_plane_client.retrieve_memory_records(**params)
                records = response.get("memoryRecordSummaries", [])
                logger.info("     ✅ Found %d relevant long-term records.", len(records))
                for record in records:
                    content = record.get("content", {}).get("text", "")
                    memory_strategy = record.get("memoryStrategyId")
                    if memory_strategy.statswith("preference"):
                        memory_records["preference"].append(content)
                    elif memory_strategy.statswith("semantic"):
                        memory_records["semantic"].append(content)
                    elif memory_strategy.statswith("summary"):
                        memory_records["summary"].append(content)
                logger.debug(f"     Retrieved records: {records}")
            
            except ClientError as e:
                logger.info("     ❌ Error querying long-term memory: %s", e)
                raise
        return [memory_records]
