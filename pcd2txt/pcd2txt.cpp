#include <string>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <iostream>
#include <pcl/io/pcd_io.h>

// global variables for table-drive
const std::string class_name[5] = {"car", "van", "pedestrian", "truck", "cyclist"};
const unsigned int label[5] = {0, 1, 2, 3, 4};
const unsigned int train_num_files[5] = {786, 528, 516, 360, 300};
const unsigned int test_num_files[5] = {336, 276, 180, 200, 204};

void 
readPCD(
        const std::string & filename,
        pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud
        )
{
    pcl::PCDReader reader;
    reader.read (filename, *cloud);
    std::cout << "readPCD(): "
                << cloud->points.size() 
                << " points in " 
                << filename
                << std::endl;
}

void 
PCD2TXT(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud,
        const unsigned int & label,
        const std::string & txt_path
        )
{
    std::ofstream writeTXT;
    writeTXT.open(txt_path.c_str(), std::ios::out);

    for (auto & p : cloud->points)
    {
        writeTXT << p.x << ' ';
        writeTXT << p.y << ' ';
        writeTXT << p.z << ' ';
        writeTXT << p.intensity << ' ';
        writeTXT << label << "\n";
    }
}


int
main(void)
{
    // path
    std::string read_file_base_path = "/media/shao/TOSHIBA EXT/data_object_velodyne/Daten";
    std::string train_pcd_path = read_file_base_path + "/train/data_augmented";
    std::string test_pcd_path = read_file_base_path + "/test/data_augmented";

    std::string train_txt_path = "/media/shao/TOSHIBA EXT/data_object_velodyne/Daten_txt_CNN/train";
    std::string test_txt_path = "/media/shao/TOSHIBA EXT/data_object_velodyne/Daten_txt_CNN/test";
    // for train data
    for (int i = 0; i != 5; ++i)
    {
        // path
        std::string train_pcd_file_path = train_pcd_path + "/" + class_name[i];

        unsigned int file_idx = 0;
        for (unsigned int j = 0; j != train_num_files[i]; ++j)
        {
            // pcd file path
            std::stringstream ss;
            ss << j;
            std::string pcd_path = train_pcd_file_path + "/" + class_name[i] + ss.str() + ".pcd";
            // read pcd data file
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
            readPCD(pcd_path, cloud);
            if (cloud->points.empty())
                continue;
            std::stringstream txt_idx;
            txt_idx << file_idx;
            ++file_idx;
            std::string txt_path = train_txt_path + "/" + class_name[i] + txt_idx.str() + ".txt";
            PCD2TXT(cloud, label[i], txt_path);
        }
    }
    // for test data
    for (int i = 0; i != 5; ++i)
    {
        // path
        std::string test_pcd_file_path = test_pcd_path + "/" + class_name[i];

        unsigned int file_idx = 0;
        for (unsigned int j = 0; j != test_num_files[i]; ++j)
        {
            // pcd file path
            std::stringstream ss;
            ss << j;
            std::string pcd_path = test_pcd_file_path + "/" + class_name[i] + ss.str() + ".pcd";
            // read pcd data file
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
            readPCD(pcd_path, cloud);
            if (cloud->points.empty())
                continue;
            std::stringstream txt_idx;
            txt_idx << file_idx;
            ++file_idx;
            std::string txt_path = test_txt_path + "/" + class_name[i] + txt_idx.str() + ".txt";
            PCD2TXT(cloud, label[i], txt_path);
        }
    }
    return 0;
}
