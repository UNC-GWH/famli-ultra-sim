#!/bin/zsh

python -m PyInstaller \
    --clean --noconfirm \
    --windowed \
    --hidden-import pkgutil \
    --hidden-import vtkmodules.all \
    --collect-data pywebvue \
    --collect-data trame_vuetify \
    --collect-data trame_vtk \
    --collect-data trame_client \
    ./annotation_main.py