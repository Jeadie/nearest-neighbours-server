import requests
from typing import List, Union, Optional
from pydantic import BaseModel

from models import IndexParams, AddItemsParams, KnnQueryParams
import urls

class HNSWIndex:
    """
    A Python client for HNSW search index API.

    Args:
        base_url (str): The base URL of the HNSW search index API.
        index_name (str): The name of the index to operate on.

    Attributes:
        base_url (str): The base URL of the HNSW search index API.
        index_name (str): The name of the index to operate on.
    """

    def __init__(self, base_url: str, index_name: str):
        """
        Constructs an instance of HNSWIndex.

        Args:
            base_url (str): The base URL of the HNSW search index API.
            index_name (str): The name of the index to operate on.
        """
        self.base_url = base_url
        self.index_name = index_name
        
        # Check if index exists and create it if it doesn't
        endpoint = f"{self.base_url}/{urls.GET_INDEX_URL}"
        response = requests.get(endpoint)
        if response.status_code == 404:
            self.create_index(IndexParams())

    def create_index(self, index_params: IndexParams) -> "HNSWIndex":
        """
        Creates a new HNSW index with the specified parameters.

        Args:
            index_params (IndexParams): An instance of the IndexParams class containing the
            parameters for the new index.

        Returns:
            dict: A dictionary containing the API response.
        """
        endpoint = f"{self.base_url}/{urls.CREATE_INDEX_URL}"
        payload = index_params.dict()
        response = requests.post(endpoint, json=payload)

        # Return a new instance of HNSWIndex pointing to the newly created index
        return HNSWIndex(self.base_url, self.index_name)

    def add_items(self, add_items_params: AddItemsParams) -> dict:
        """
        Adds new items to an existing HNSW index.

        Args:
            add_items_params (AddItemsParams): An instance of the AddItemsParams class containing
            the items to add to the index.

        Returns:
            dict: A dictionary containing the API response.
        """
        endpoint = f"{self.base_url}/{urls.ADD_ITEMS_URL}"
        payload = add_items_params.dict()
        response = requests.post(endpoint, json=payload)
        return response.json()

    def mark_deleted(self, label: int) -> dict:
        """
        Marks an item in the HNSW index as deleted.

        Args:
            label (int): The label of the item to mark as deleted.

        Returns:
            dict: A dictionary containing the API response.
        """
        endpoint = f"{self.base_url}/{urls.DELETE_URL}"
        payload = {"label": label}
        response = requests.post(endpoint, json=payload)
        return response.json()

    def unmark_deleted(self, label: int) -> dict:
        """
        Unmarks an item in the HNSW index as deleted.

        Args:
            label (int): The label of the item to unmark as deleted.

        Returns:
            dict: A dictionary containing the API response.
        """
        endpoint = f"{self.base_url}/{urls.UNDELETE_URL}"
        payload = {"label": label}
        response = requests.post(endpoint, json=payload)
        return response.json()

    def resize_index(self, new_size: int) -> dict:
        """
        Resizes the HNSW index to a new maximum size.

        Args:
            new_size (int): The new maximum size of the index.

        Returns:
            dict: A dictionary containing the API response.
        """
        endpoint = f"{self.base_url}/{urls.RESIZE_INDEX_URL}"
        payload = {"new_size": new_size}
        response = requests.post(endpoint, json=payload)
        return response.json()

    def set_ef(self, ef: int) -> dict:
        """
        Sets the ef parameter of the index on the server.

        Args:
            ef (int): The new value of the ef parameter.

        Returns:
            dict: The server response.
        """
        endpoint = f"{self.base_url}/{urls.SET_EF_URL}"
        payload = {"ef": ef}
        response = requests.post(endpoint, json=payload)
        return response.json()

    def knn_query(self, knn_query_params: KnnQueryParams, k: int = 1) -> dict:
        """
        Queries the index for the k nearest neighbors to the provided query element.

        Args:
            knn_query_params (KnnQueryParams): The query element and its parameters.
            k (int, optional): Number of nearest neighbors to search for. Defaults to 1.

        Returns:
            dict: A dictionary containing the labels and distances of the k nearest neighbors.
        """
        endpoint = f"{self.base_url}/{urls.KNN_QUERY_URL}"
        payload = knn_query_params.dict()
        payload["k"] = k
        response = requests.post(endpoint, json=payload)
        return response.json()

    def load_index(self, max_elements: int = 0, allow_replace_deleted: bool = False) -> dict:
        """
        Loads the index from disk.

        Args:
            max_elements (int, optional): Maximum number of elements to load from disk. If set to 0, all elements are loaded.
            allow_replace_deleted (bool, optional): If set to True, elements that have been marked as deleted will be replaced.

        Returns:
            dict: The response from the server.

        """
        endpoint = f"{self.base_url}/{urls.LOAD_INDEX_URL}"
        payload = {"max_elements": max_elements, "allow_replace_deleted": allow_replace_deleted}
        response = requests.post(endpoint, json=payload)
        return response.json()

    def save_index(self) -> dict:
        """
        Saves the index to disk.

        Returns:
            dict: The response from the server.

        """
        endpoint = f"{self.base_url}/{urls.SAVE_INDEX_URL}"
        response = requests.post(endpoint)
        return response.json()
