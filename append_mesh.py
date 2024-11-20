import os
import argparse
import vtk
import sys 

def merge_meshes_in_directory(args):
    # Initialize AppendPolyData
    append_filter = vtk.vtkAppendPolyData()

    # Supported file extensions
    extensions = ['.stl', '.obj', '.vtk']
    
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(args.dir):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in extensions:
                file_path = os.path.join(root, filename)
                print(f"Processing {file_path}...")

                if filename.lower().endswith('.stl'):
                    reader = vtk.vtkSTLReader()
                elif filename.lower().endswith('.obj'):
                    reader = vtk.vtkOBJReader()
                elif filename.lower().endswith('.vtk'):
                    reader = vtk.vtkPolyDataReader()
                else:
                    # Unsupported file type
                    continue

                reader.SetFileName(file_path)
                reader.Update()

                # Append the current mesh
                append_filter.AddInputData(reader.GetOutput())

    append_filter.Update()

    if args.decimate > 0.0:
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(append_filter.GetOutput())
        decimate.SetTargetReduction(args.decimate)
        decimate.Update()

        append_filter = vtk.vtkAppendPolyData()
        append_filter.AddInputData(decimate.GetOutput())
        append_filter.Update()

    # Output merged mesh
    if args.out.lower().endswith('.stl'):
        writer = vtk.vtkSTLWriter()
    elif args.out.lower().endswith('.obj'):
        writer = vtk.vtkOBJWriter()
    elif args.out.lower().endswith('.vtk'):
        writer = vtk.vtkPolyDataWriter()
    else:
        # Unsupported file type
        print("Unsupported file type:", args.out, file=sys.stderr)
        return
    
    writer.SetFileName(args.out)
    writer.SetInputData(append_filter.GetOutput())
    writer.Write()

    print(f"All meshes merged and saved to {args.out}")

def main():
    parser = argparse.ArgumentParser(description='Merge STL, OBJ, and VTK files in a directory into a single STL file.')
    parser.add_argument('--dir', type=str, help='Directory containing the mesh files to merge.')
    parser.add_argument('--decimate', type=float, help='Use decimate pro filter to reduce the number of triangles in the mesh. This is the target reduction (default: 0.0 or no reduction)', default=0.0)
    parser.add_argument('--out', type=str, help='Output filename', default='out.stl')
    
    args = parser.parse_args()

    merge_meshes_in_directory(args)

if __name__ == "__main__":
    main()
