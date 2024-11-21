import os
import sys
import numpy as np
import pypatchworkpp

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, '../../data/')

try:
    patchwork_module_path = os.path.join(cur_dir, "../../build/python_wrapper")
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)

def read_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    return scan

if __name__ == "__main__":

    # Patchwork++ initialization
    params = pypatchworkpp.Parameters()
    params.verbose = True

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    for file in sorted(os.listdir(data_dir)):

        # Load point cloud
        pointcloud = read_bin(data_dir+file)
        pointcloud = np.concatenate((pointcloud, np.arange(pointcloud.shape[0])), axis=1)
        print(pointcloud.shape)

        # Estimate Ground
        PatchworkPLUSPLUS.estimateGround(pointcloud)

        # Get Ground and Nonground
        ground      = PatchworkPLUSPLUS.getGround()
        nonground   = PatchworkPLUSPLUS.getNonground()
        time_taken  = PatchworkPLUSPLUS.getTimeTaken()

        # Get centers and normals for patches
        centers     = PatchworkPLUSPLUS.getCenters()
        normals     = PatchworkPLUSPLUS.getNormals()

        print("Origianl Points  #: ", pointcloud.shape[0])
        print("Ground Points    #: ", ground.shape[0])
        print("Nonground Points #: ", nonground.shape[0])
        print("Time Taken : ", time_taken / 1000000, "(sec)")
        print("Press ... \n")
        print("\t H  : help")
        print("\t N  : visualize the surface normals")
        print("\tESC : close the Open3D window")

