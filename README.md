# SD-UNET
A Novel Segmentation Framework for CT Images of Lung Infections

# REQUIREMENTS
window10(Ubuntu is Also OK)+pycharm+python3.7+pytorch1.8.1

# RUN:
Firstly, you should do is download the datasets and create the path of the datasets. then run ~ example:
python main.py --action train&test --Model SD-UNet --epoch 200 --batch_size 4
 
# RESULTS
model/model-3 folder:
After training, the saved model of binary-class and multi-class segmentation is in this folder.
result/result-3 folder:
After training/test, the saved evaluation reslut of binary-class and multi-class segmentation is in this folder.

# the datasets
1.COVID-19 CT segmentation dataset. Accessed: Apr. 11, 2020. [Online]. Available: https://medicalsegmentation.com/covid19/
2.Ma, J.; Ge, C.; Wang, Y.; An, X.; Gao, J.; Yu, Z. COVID-19 CT lung and infection segmentation dataset. 2020. [Online]. Available: https://doi.org/10.5281//zenodo. 3757476
