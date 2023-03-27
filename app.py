from typing import List, Optional, Union
import hnswlib
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import os
from apscheduler.schedulers.background import BackgroundScheduler

import urls
from models import IndexParams, AddItemsParams, KnnQueryParams

app = FastAPI()

scheduler = BackgroundScheduler()

@app.on_event("startup")
async def start_scheduler():
    scheduler.start()


# Set a base directory to store the indexes
BASE_DIR = "./indexes"


@app.post(urls.CREATE_INDEX_URL)
def create_index(index_name: str, index_params: IndexParams, background_tasks: BackgroundTasks):
    """Creates a new HNSW index with the given space and dimensionality.

    Args:
        index_name (str): Name of the index to be created.
        index_params (IndexParams): Parameters for initializing the index.
        background_tasks (BackgroundTasks): BackgroundTasks instance to schedule a task.

    Returns:
        Dict: Response message indicating whether the index was created successfully.
    """
    index = hnswlib.Index(index_params.space, index_params.dim)
    index.init_index(index_params.max_elements, index_params.M, index_params.ef_construction, index_params.random_seed,
                     index_params.allow_replace_deleted)
    app.state.indexes[index_name] = index
    scheduler.add_job(save_index, 'cron', args=index_name, name= f"save_{index_name}", hour='*', minute=0)
    
    return {"message": f"Index {index_name} created and initialized."}

@app.get(urls.GET_INDEX_URL)
def get_index(index_name: str):
    """
    Retrieves the specified HNSW index.

    Args:
        index_name (str): The name of the index to retrieve.

    Returns:
        dict: A dictionary containing the retrieved index.
    """
    load_index(index_name, 0, allow_replace_deleted=False)
    index = app.state.indexes.get(index_name)

    properties = {
        "space": index.space,
        "dim": index.dim,
        "M": index.M,
        "ef_construction": index.ef_construction,
        "max_elements": index.max_elements,
        "element_count": index.element_count,
        "ef": index.ef,
        "num_threads": index.num_threads
    }
    return properties

@app.post(urls.ADD_ITEMS_URL)
def add_items(index_name: str, add_items_params: AddItemsParams):
    """Inserts data into the index.

    Args:
        add_items_params (AddItemsParams): Parameters for adding items to the index.
        index_name (str): Name of the index to which data is to be added.
    """
    index = app.state.indexes.get(index_name)
    data = [[float(x) for x in item.data] for item in add_items_params.items]
    data_arr = hnswlib.IndexData()
    for item in add_items_params.items:
        data_arr.add_data(data[item.id], item.id)
    index.add_items(data_arr, add_items_params.num_threads, add_items_params.replace_deleted)
    return {"message": f"{len(data)} items added to index {index_name}."}

@app.post(urls.DELETE_URL)
def delete(label: int, index_name: str):
    """Marks an element as deleted

    Args:
        label (int): The integer label of the element to mark as deleted.
        index_name (str): Name of the index from which the element is to be marked as deleted.
    """
    index = app.state.indexes.get(index_name)
    index.mark_deleted(label)
    return {"message": f"Item with label {label} marked as deleted in index {index_name}."}

@app.post(urls.UNDELETE_URL)
def undelete(label: int, index_name: str):
    """Unmarks an element as deleted.

    Args:
        label (int): The integer label of the element to unmark as deleted.
        index_name (str): Name of the index from which the element is to be unmarked as deleted.
    """
    index = app.state.indexes.get(index_name)
    index.unmark_deleted(label)
    return {"message": f"Item with label {label} unmarked as deleted in index {index_name}."}

@app.post(urls.RESIZE_INDEX_URL)
def resize_index(new_size: int, index_name: str):
    """Changes the maximum capacity of the index.

    Args:
        new_size (int): The new maximum capacity of the index.
        index_name (str): Name of the index to be resized.
    """
    index = app.state.indexes.get(index_name)
    index.resize_index(new_size)
    return {"message": f"Index {index_name} resized to {new_size}."}

@app.post(urls.SET_EF_URL)
def set_ef(ef: int, index_name: str):
    """Sets the query time accuracy/speed trade-off.

    Args:
        ef (int): The query time accuracy/speed trade-off.
        index_name (str): Name of the index to which the trade-off is to be set.
    """
    index = app.state.indexes.get(index_name)
    index.set_ef(ef)
    return {"message": f"Index {index_name} set to query time accuracy/speed trade-off of {ef}."}


@app.post(urls.KNN_QUERY_URL)
def knn_query(index_name: str, knn_query_params: KnnQueryParams):
    """Queries the index for the k closest elements for each element of the data.

    Args:
        knn_query_params (KnnQueryParams): Parameters for the k nearest neighbors query.
        index_name (str): Name of the index to be queried.

    Returns:
        Dict: Dictionary containing a list of the k closest element labels for each element of data, and a list of corresponding distances.
    """
    index = app.state.indexes.get(index_name)
    data = [[float(x) for x in item] for item in knn_query_params.data]
    data_arr = hnswlib.IndexData()
    for d in data:
        data_arr.add_data(d)
    query_labels, distances = index.knn_query(data_arr, knn_query_params.k, knn_query_params.num_threads, knn_query_params.filter)
    return {"query_labels": query_labels.tolist(), "distances": distances.tolist()}


@app.post(urls.LOAD_INDEX_URL)
def load_index(index_name: str, max_elements: int = 0, allow_replace_deleted: bool = False):
    """Loads the index from persistence to the uninitialized index.

    Args:
        max_elements (int, optional): Maximum number of elements in the structure. Defaults to 0.
        allow_replace_deleted (bool, optional): Whether the index being loaded has enabled replacing of deleted elements. Defaults to False.
        index_name (str): Name of the index to which the loaded index is to be assigned.

    Returns:
        Dict: Dictionary containing a message indicating whether the index was successfully loaded.
    """
    path_to_index = os.path.join(BASE_DIR, index_name)
    index = hnswlib.Index(None, None)
    index.load_index(path_to_index, max_elements, allow_replace_deleted)
    app.state.indexes[index_name] = index
    return {"message": f"Index {index_name} loaded from {path_to_index}."}


@app.post(urls.SAVE_INDEX_URL)
def save_index(index_name: str):
    """Saves the index to persistence.

    Args:
        index_name (str): Name of the index to be saved.

    Returns:
        Dict: Dictionary containing a message indicating whether the index was successfully saved.
    """
    path_to_index = os.path.join(BASE_DIR, index_name)
    index = app.state.indexes.get(index_name)
    index.save_index(path_to_index)
    return {"message": f"Index {index_name} saved to {path_to_index}."}


@app.post(urls.SET_NUM_THREADS_URL)
def set_num_threads(num_threads: int, index_name: str):
    """Sets the default number of cpu threads used during data insertion/querying.
    Args:
        num_threads (int): The default number of cpu threads used during data insertion/querying.
        index_name (str): Name of the index to which the default number of threads is to be set.
    """
    index = app.state.indexes.get(index_name)
    index.set_num_threads(num_threads)
    return {"message": f"Index {index_name} set to use {num_threads} threads."}


@app.get(urls.GET_ITEMS_URL)
def get_items(ids: List[int], index_name: str):
    """Returns vectors that have integer identifiers specified in ids.

    Args:
        ids (List[int]): List of integer identifiers.
        index_name (str): Name of the index from which to retrieve vectors.

    Returns:
        List[List[float]]: List of vectors that have integer identifiers specified in ids.
    """
    index = app.state.indexes.get(index_name)
    items = index.get_items(ids)
    items = [[float(x) for x in item] for item in items]
    return {"items": items}

@app.get(urls.GET_IDS_LIST_URL)
def get_ids_list(index_name: str):
    """Returns a list of all elements' ids.

    Args:
        index_name (str): Name of the index to which the list of ids is to be retrieved.

    Returns:
        List[int]: List of all elements' ids.
    """
    index = app.state.indexes.get(index_name)
    ids_list = index.get_ids_list()
    return {"ids_list": ids_list}

@app.get(urls.GET_MAX_ELEMENTS_URL)
def get_max_elements(index_name: str):
    """Returns the current capacity of the index.
    Args:
        index_name (str): Name of the index from which the capacity is to be retrieved.

    Returns:
        int: The current capacity of the index.
    """
    index = app.state.indexes.get(index_name)
    max_elements = index.get_max_elements()
    return {"max_elements": max_elements}

@app.get(urls.GET_CURRENT_COUNT_URL)
def get_current_count(index_name: str):
    """Returns the current number of element stored in the index.

    Args:
        index_name (str): Name of the index from which the current number of elements is to be retrieved.

    Returns:
        int: The current number of element stored in the index.
    """
    index = app.state.indexes.get(index_name)
    current_count = index.get_current_count()
    return {"current_count": current_count}

@app.post(urls.SCHEDULE_URL)
async def schedule(background_tasks: BackgroundTasks, index_name: str, schedule_interval: int):
    """Schedules a background task to save the given index to disk at a specified interval.

    Args:
        background_tasks (BackgroundTasks): FastAPI background task manager.
        index_name (str): Name of the index to be saved.
        schedule_interval (int): Interval, in hours, at which the index should be saved.

    Returns:
        Dict: Response message indicating whether the background task was scheduled successfully.
    """
    scheduler.add_job(save_index, 'cron', replace_existing=True, args=index_name, name= f"save_{index_name}", hour='*/{}'.format(schedule_interval), minute=0)
    return {"message": f"Background task scheduled to save index {index_name} every {schedule_interval} hour(s)."}
