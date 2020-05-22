# keras_tests
Some projects using Keras for learning and classification tasks


# Dog/cat breed classification 
Some inspiration of this code is based on article https://medium.com/@iliazaitsev/dogs-breeds-classification-with-keras-b1fd0ab5e49c

The actual solution works in 2 steps:
1. Pretrained Yolo v3 [1] image detection model is used to find cats and dogs in the image.
2. Depending on the animal found in the image, the bounding box is passed to the breed classifier (there are different classifiers for different animals) that returns 3 most probable breeds of that animal.
 
Each of these classifiers is based on the Xception [2] model pretrained on the Imagenet [3] was fine-tuned on datasets with cat and dog breeds.
The system can be directly deployed. When the `FullEvaluator` started, it can automatically load the saved models - animal detector and breed classifiers - and is prepared to classify images it is given.

## Some Results
Correct classifications: 
<tr>
    <td> <img src="dog_cat_breeds/results/Bengal_110_detected.jpg" alt="Drawing" style="width: 100px;"/> </td>
    <td> <img src="dog_cat_breeds/results/wolker_hound_1412_detected.jpg" alt="Drawing" style="width: 150px;"/> </td>
    <td> <img src="dog_cat_breeds/results/bloodhound_10309_detected.jpg" alt="Drawing" style="width: 150px;"/> </td>
</tr>

Incorrect classifications: 
<tr>
    <td> <img src="dog_cat_breeds/results/Bengal_101_detected_wrongEgypt.jpg" alt="Drawing" style="width: 250px;"/> </td>
    <td> <img src="dog_cat_breeds/results/Birman_125_detected_wrongRagdoll.jpg" alt="Drawing" style="width: 250px;"/> </td>
</tr>

![Cat1](dog_cat_breeds/results/Bengal_101_detected_wrongEgypt.jpg)|![Dog1](dog_cat_breeds/results/Birman_125_detected_wrongRagdoll.jpg))

[1] Redmon, J. and Farhadi, A., 2018. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.

[2] Chollet, F., 2017. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1251-1258).

[3] Deng, J., Dong, W., Socher, R., Li, L.J., Li, K. and Fei-Fei, L., 2009, June. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition (pp. 248-255). Ieee.
