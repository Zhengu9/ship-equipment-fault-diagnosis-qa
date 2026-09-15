import json
from pymilvus import MilvusClient, DataType


class VectorStore:
    def __init__(self, name):
        self.client = MilvusClient(uri="http://localhost:19530")
        self.name = name
        self._load()
        
    def _load(self):
        if self.name not in self.client.list_collections():
            print(f"Creating collection {self.name}")
            
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)

            self.client.create_collection(
                collection_name=self.name,
                schema=schema
            )
            
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="L2",
                index_type="HNSW",
                index_name="vector_index",
                params={
                    "M": 64,
                    "efConstruction": 512
                },
            )

            self.client.create_index(
                collection_name=self.name,
                index_params=index_params,
                sync=True # Whether to wait for index creation to complete before returning. Defaults to True.
            )
            
        index_details = self.client.describe_index(
            collection_name=self.name,
            index_name="vector_index"
        )
        print(f"Index details: {index_details}")
        
        self.client.load_collection(self.name)
        print(f"Collection {self.name} loaded")
        
    def insert(self, data):
        """
        data = [
            {"id": 0, "vector": [0.1]*768, "name": "test0"},
            {"id": 1, "vector": [0.2]*768, "name": "test1"},
        ]
        """
        self.client.insert(
            collection_name=self.name, 
            data=data
        )
        
    def search(self, data, top_k):
        """
        data = [
            [0] *768,
            [1] *768,
        ]
        """
        results = self.client.search(
            collection_name=self.name,
            data=data,
            limit=top_k,
            search_params={
                "metric_type": "L2", 
                "params": {
                    "efSearch": top_k*8
                }
            },
            output_fields=["name"]
        )
        results = json.loads(json.dumps(results))
        return [sorted(result, key=lambda x: x['distance']) for result in results]
    
    def get(self, ids):
        """
        ids = [0, 1]
        """
        resp = self.client.get(
            collection_name=self.name,
            ids=ids,
            output_fields=["name", "vector"]
        )
        results = []
        for each in resp:
            results.append(each)
        return results
    
    def count(self):
        result = self.client.query(
            collection_name=self.name,
            output_fields=["count(*)"]
        )
        for each in result:
            return each['count(*)']
        
    def reset(self):
        self.client.drop_collection(
            collection_name=self.name
        )
        self._load()