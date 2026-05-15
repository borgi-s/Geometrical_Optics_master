"""I/O helpers: image read/write, strain-field caching, video export."""

import os


def check_folder(path: str, folder_name: str) -> None:
    """Create `path/folder_name` if it does not exist. Print a status message."""
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created.")
    else:
        print(f"Folder {folder_path} already exists.")
