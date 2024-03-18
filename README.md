**Generate motion blur image:**  
  motion_blur file
  File placement：  
    input_image(Input HR image)  
    output_image(Generates LR image)  
    ```
    python blur_image.py   
    ```   

**Tain correct motion blur model:**  
  File placement：  
    Tain_hr(Input HR image)  
    Tain_lr(Input LR image)  
    val_hr (Input HR image)  
    val_lr (Input LR image)  
    ```
    python convert.py   
    ```  
    ```
    python main.py
    ```  

**Predict the correct motion blur model：**  
    test (The file input test image)  
    outputgan (The file automatically generates the correct image)  
    ```
    python pred.py  
    ```  

**Test recognition result:**  
    use the CNN_for_ESRGAN.py  

**Weight provide:**  
    ESRGAN：https://drive.google.com/drive/folders/1Yfune8oilz5U9OD2hnZigUudAZ2SKwd4?usp=sharing  
    CNN：https://drive.google.com/drive/folders/1LjtR-lSSSJXQaO2LP8UWrlJ390XL6FB0?usp=sharing

**SIF methods (Image similarity)**  
    use the similarity_SIFT.py   

**Acknowledgments**
Code borrows heavily from ![ESRGAN](https://github.com/xinntao/ESRGAN.git) and DeblurGan 
