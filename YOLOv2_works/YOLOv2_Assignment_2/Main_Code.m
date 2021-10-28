%% Object Detection Using YOLO v2 Object Detector
% The following code demonstrates running prediction on a pre-trained YOLO v2 
% network, trained on COCO dataset.

% CLeanup previous data/variables:
clc;
clear all;
close all;
clear functions;
%% *Download the pre-trained network*

modelName = 'tinyYOLOv2-coco';
helper.downloadPretrainedYOLOv2(modelName);

pretrained = load(modelName);
detector = pretrained.yolov2Detector;

%setup:
%     filepath to folder testing_dataset
file_path_dataset = append(pwd,'\','testing_dataset\')
d = dir(append(file_path_dataset,'*.jpg'));
images_count_in_folder = length(d);
need_read = cell(1,images_count_in_folder);
save_file_path = append(pwd,'\testing_results\');

% Detect Objects using YOLO v2 Object Detector
% Read test image.
% img = imread('sherlock.jpg');
for n=1:images_count_in_folder
    img{n} = imread(append(file_path_dataset,d(n).name));

    % Detect objects in test image.
    [boxes, scores, labels] = detect(detector, img{n});

    % Visualize detection results.
    img{n} = insertObjectAnnotation(img{n},'rectangle',boxes,labels);
    % figure(n), imshow(img{n}) % laggy when running lots of data if uncomment :))
    imwrite(img{n},[save_file_path,d(n).name],'jpg');
end


