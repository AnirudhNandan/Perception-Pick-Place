# Project: Perception Pick & Place

## Goal:
To recognize items in 3 seperate pick lists and generate appropriate robot motion instructions.

## Training SVM models:

#### Capturing features:
Visual feature data was created for the following objects:

Code:
```python
if __name__ == '__main__':
    rospy.init_node('capture_node')

    models = [\
       'sticky_notes',
       'book',
       'snacks',
       'biscuits',
       'eraser',
       'soap2',
       'soap',
       'glue']
```

Fifteen attempts are made to get valid point cloud in various orientations.

Code:
```python
for model_name in models:
        spawn_model(model_name)

        for i in range(15):
            # make 15 attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 15:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])
```

#### SVM model:

A SVM model was trained to recognize objects. High "C" value was selected here to get complex decision boundary. It was found "sigmoid" model worked better for this project. In this project the goal is to classify a certain know objects and no new objects are expected while testing/putting model to work. Hence, high C is desirable and we can get away with it. This will not apply if say there is a test object which is a book but looks little different from training set book then lower C value will be useful to get less complex decision boundary which can generalize models.

Code:
```python
clf = svm.SVC(C=9.3, kernel='sigmoid', gamma='auto')
```

![model_1](https://user-images.githubusercontent.com/7349926/34089159-52725114-e37c-11e7-8ffd-b24ed9ff0303.png)



![model_2](https://user-images.githubusercontent.com/7349926/34089166-6394c33c-e37c-11e7-8595-74b74dc803df.png)



![model_3](https://user-images.githubusercontent.com/7349926/34089184-71126e06-e37c-11e7-8b61-02cc583431d8.png)

## Subscribing to ROS node:

A subscriber to receive point cloud data from the camera is created and assigned to "pcl_sub". Various publishers are created to publish manipulated point clound data, detected objects, and object markers.

Code:
```python
# TODO: ROS node initialization
rospy.init_node('clustering', anonymous=True)
# TODO: Create Subscribers
pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
# TODO: Create Publishers
pcl_pub = rospy.Publisher("/points", PointCloud2, queue_size=1)
pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
pcl_clusters_pub = rospy.Publisher("/pcl_clusters", PointCloud2, queue_size=1)
object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
```
## Applying Pass Through filter:

First of all a function called "pcl_callback" is created which will process point clound data.

Code:
```python
def pcl_callback(pcl_msg):

```

The objects of interest are on the table hence a passthrough filter to filter out table is required. Further to remove pointcloud of boxes, a passthrough filter in y axis is implemented.

Code:
```python
# Exercise-2 TODOs:
# TODO: Convert ROS msg to PCL data
pcl_msg = ros_to_pcl(pcl_msg)
# TODO: Statistical Outlier Filtering
# TODO: Voxel Grid Downsampling
vox = pcl_msg.make_voxel_grid_filter()
LEAF_SIZE = 0.009 
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
cloud_filtered = vox.filter()
# TODO: PassThrough Filter
passthrough = cloud_filtered.make_passthrough_filter()
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.61
axis_max = 0.8
passthrough.set_filter_limits(axis_min, axis_max)
cloud_filtered = passthrough.filter()
# Filter Y axis
passthrough = cloud_filtered.make_passthrough_filter()
filter_axis = 'y'
passthrough.set_filter_field_name(filter_axis)
axis_min = -0.55
axis_max = 0.55
passthrough.set_filter_limits(axis_min, axis_max)
cloud_filtered = passthrough.filter()
```

## Applying RANSAC filter:

Using a RANSAC filter table and objects are seperated.

Code:
```python
# TODO: RANSAC Plane Segmentation
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
max_distance = 0.001
seg.set_distance_threshold(max_distance)
inliers, coefficients = seg.segment()
# TODO: Extract inliers and outliers
cloud_table = cloud_filtered.extract(inliers, negative=False)
cloud_objects = cloud_filtered.extract(inliers, negative=True)
```

## Applying Euclidean clustering:

By using Euclidean clustering point cloud is clustered into distinct groups.

Code:
```python
# TODO: Euclidean Clustering
white_cloud = XYZRGB_to_XYZ(cloud_objects)
tree = white_cloud.make_kdtree()
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.01)
ec.set_MinClusterSize(50)
ec.set_MaxClusterSize(900)
ec.set_SearchMethod(tree)
cluster_indices = ec.Extract()
```

Clusters are visualized by assigning colors to each cluster. Manipulated point cloud data are published to various publishers.

Code:
```python
# TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
cluster_color = get_color_list(len(cluster_indices))
color_cluster_point_list = []

for j, indices in enumerate(cluster_indices):
    for i, indice in enumerate(indices):
		color_cluster_point_list.append([white_cloud[indice][0],
		                                    white_cloud[indice][1],
		                                    white_cloud[indice][2],
		                                     rgb_to_float(cluster_color[j])])

cluster_cloud = pcl.PointCloud_PointXYZRGB()
cluster_cloud.from_list(color_cluster_point_list)
# TODO: Convert PCL data to ROS messages
ros_cloud = pcl_to_ros(cloud_filtered)
ros_cloud_objects = pcl_to_ros(cloud_objects)
ros_cloud_table = pcl_to_ros(cloud_table)
ros_cluster_cloud = pcl_to_ros(cluster_cloud)
# TODO: Publish ROS messages
pcl_pub.publish(ros_cloud)
pcl_objects_pub.publish(ros_cloud_objects)
pcl_table_pub.publish(ros_cloud_table)
pcl_clusters_pub.publish(ros_cluster_cloud)
```

## Object recognition:

By using prediction model which was trained offline the detected clusters are classified.

Code:
```python
# Exercise-3 TODOs:
detected_objects_labels = []
detected_objects = []
# Classify the clusters! (loop through each detected cluster one at a time)
for index, pts_list in enumerate(cluster_indices):
    # Grab the points for the cluster
	pcl_cluster = cloud_objects.extract(pts_list)
	ros_cluster = pcl_to_ros(pcl_cluster)
	# Compute the associated feature vector
	chists = compute_color_histograms(ros_cluster, using_hsv=True)
	normals = get_normals(ros_cluster)
	nhists = compute_normal_histograms(normals)
	feature = np.concatenate((chists, nhists))
	# Make the prediction
	prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
	label = encoder.inverse_transform(prediction)[0]
	detected_objects_labels.append(label)
	# Publish a label into RViz
	label_pos = list(white_cloud[pts_list[0]])
	label_pos[2] += .4
	object_markers_pub.publish(make_label(label,label_pos, index))
	# Add the detected object to the list of detected objects.
	do = DetectedObject()
	do.label = label
	do.cloud = ros_cluster
    detected_objects.append(do)
# Publish the list of detected objects
rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
detected_objects_pub.publish(detected_objects)
```

## Calculate centroid:

The program loops over the object list and calculates centroids of detected clusters.

The following code to centroid was obtained from discussion in slack udacity_perception channel from "reno" and "tokyo_adam"

Code:
```python
# TODO: Initialize variables
test_scene_num = Int32()
object_name = String()
arm_name = String()
pick_pose = Pose()
place_pose = Pose()

dict_list = []
test_scene_num.data = 2

labels = []
centroids = []
# TODO: Get/Read parameters
object_list_param = rospy.get_param('/object_list')
dropbox_param = rospy.get_param('/dropbox')
# TODO: Parse parameters into individual variables
for obj in object_list:
    labels.append(obj.label)
    points_arr = ros_to_pcl(obj.cloud).to_array()
    centroids.append(np.mean(points_arr, axis=0)[:3])
```
The centroid of detected cluster is converted to native python scalar type as ROS only supports native python scalar type. The [x,y,z] co-ordinates of centroids are assigned to "pick_pose.position" will be used to initiate pick-up.

Code:
```python
for i in range(0, len(object_list_param)):
    	object_name.data = object_list_param[i]['name']
    	object_group = object_list_param[i]['group']
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        for j in range(0,len(labels)):
        	if object_name.data == labels[j]:
        		pick_pose.position.x = np.asscalar(centroids[j][0])
        		pick_pose.position.y = np.asscalar(centroids[j][1])
        		pick_pose.position.z = np.asscalar(centroids[j][2])
```

## Creating output YAML files:

A dictionary of "test_scene_num", "arm_name", "object_name", "pick_pose", and "place_pose" data is created with in pick list loop. The whole dictionary is written to a yaml file outside the loop.
Code:
```python
for i in range(0, len(object_list_param)):
....
....
....
    for i in range(0, len(object_list_param)):
        	yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
    		dict_list.append(yaml_dict)
            
yaml_filename = 'output_1.yaml'
send_to_yaml(yaml_filename, dict_list)
```

## Results:
Detecting items in pick list 1: All items were correctly detected and recognized (100%). Pick-up instructions were generated and stored in output_1.yaml
![screen shot 2017-12-17 at 11 01 43 pm](https://user-images.githubusercontent.com/7349926/34089497-77909f26-e37e-11e7-9611-5722e061d5e3.png)

Detecting items in pick list 2: All items were correctly detected and recognized (100%). Pick-up instructions were generated and stored in output_2.yaml
![screen shot 2017-12-17 at 9 49 12 pm](https://user-images.githubusercontent.com/7349926/34089227-acc69d46-e37c-11e7-8ef5-5b165fcb359f.png)

Detecting items in pick list 3: All items were correctly detected and recognized (100%). Pick-up instructions were generated and stored in output_3.yaml
![screen shot 2017-12-17 at 9 53 52 pm](https://user-images.githubusercontent.com/7349926/34089238-b8dc12e6-e37c-11e7-991b-aa8039dce9d9.png)

## Enclosed files:

1. perception_project.py: This file contains "pcl_callback" and "pr2_mover" functions.
2. output_1.yaml: PickPlace request parameters for list 1
3. output_2.yaml: PickPlace request parameters for list 2
4. output_3.yaml: PickPlace request parameters for list 3
