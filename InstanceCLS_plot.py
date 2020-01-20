from utils.PlotConfusion import plot_confusion_matrix

# Plot the matrix using Py3Torch04
model_arch = ['squeezenetv2', 'shufflenetv2', 'mobilenetv2',
              'vggnet16', 'resnet50', 'densenet121']
title_name = ['(a) SqueezeNet', '(b) ShuffleNetV2', '(c) MobileNetV2',
              '(d) VGGNet16', '(e) ResNet50', '(f) DenseNet121']
# for i in range(5):
i=5
plot_confusion_matrix("results/%s_truth_label.txt"%(model_arch[i]),
                      "results/%s_pred_label.txt"%(model_arch[i]),
                      "results/%d_%s_confusion_matirx.png"%(i+1,model_arch[i]),
                      title_name[i])
