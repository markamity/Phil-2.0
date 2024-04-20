import pinecone
from typing import Optional, List, Iterator, Dict, Tuple, cast
import uuid
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.utils import printd, datetime_to_timestamp, timestamp_to_datetime
from memgpt.config import MemGPTConfig
from memgpt.data_types import Record, Message, Passage, RecordType

class PineconeStorageConnector(StorageConnector):
    """Storage via Pinecone"""

    def __init__(self, table_type: str, config: MemGPTConfig, user_id, agent_id=None):
        super().__init__(table_type=table_type, config=config, user_id=user_id, agent_id=agent_id)
        self.table_name = "your_index_name"  # Adjust index name as needed

        # Connect to Pinecone
        pinecone.init(api_key=config.pinecone_api_key)  # Assuming you have an API key

        # Create or get the index
        self.index = pinecone.Index(self.table_name)

    def format_records(self, records: List[RecordType]):
        # Assuming embeddings are already generated for records
        ids = [str(record.id) for record in records]
        embeddings = [record.embedding for record in records]

        # Format metadata if available
        metadatas = []
        for record in records:
            metadata = vars(record)
            metadata.pop("id")
            metadata.pop("text")
            metadata.pop("embedding")
            if "created_at" in metadata:
                metadata["created_at"] = datetime_to_timestamp(metadata["created_at"])
            if "metadata_" in metadata and metadata["metadata_"] is not None:
                record_metadata = dict(metadata["metadata_"])
                metadata.pop("metadata_")
            else:
                record_metadata = {}
            metadata = {key: value for key, value in metadata.items() if value is not None}  # Remove null values
            metadata = {**metadata, **record_metadata}  # Merge with metadata
            metadatas.append(metadata)

        return ids, embeddings, metadatas

    def insert(self, record: Record):
        ids, embeddings, metadatas = self.format_records([record])
        self.index.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(self, query_vec: List[float], top_k: int = 10) -> List[RecordType]:
        results = self.index.query(queries=query_vec, top_k=top_k)
        # Convert results to your RecordType
        return self.results_to_records(results)

    def query_date_range(self, start_date, end_date, top_k: int = 10) -> List[RecordType]:
        # Convert date range to timestamp
        start_timestamp = datetime_to_timestamp(start_date)
        end_timestamp = datetime_to_timestamp(end_date)

        # Construct query filter
        query_filter = {
            "created_at": {"$gte": start_timestamp, "$lte": end_timestamp}
        }

        results = self.index.query(query_filter=query_filter, top_k=top_k)
        return self.results_to_records(results)

    def get_page(self, page_num: int, page_size: int) -> List[RecordType]:
        # Calculate offset based on page number and page size
        offset = (page_num - 1) * page_size

        # Query for a specific page
        results = self.index.query(offset=offset, limit=page_size)
        return self.results_to_records(results)

    def delete(self, record_id: uuid.UUID):
        self.index.delete(ids=[str(record_id)])

    def get(self, id: uuid.UUID) -> Optional[RecordType]:
        results = self.index.query(ids=[str(id)])
        if results:
            return self.results_to_records(results)[0]
        else:
            return None

    def delete_all(self):
        self.index.delete_all()

    def save(self):
        # Save index (optional for Pinecone)
        self.index.save()

    def size(self) -> int:
        return self.index.size()

    def list_indexes(self) -> List[str]:
        return pinecone.list_indexes()

    def get_index_info(self) -> Dict:
        return self.index.info()

    # Additional methods can be added based on your needs
