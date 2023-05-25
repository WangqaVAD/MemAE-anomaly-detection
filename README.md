# MemAE-anomaly-detection
**Model reference paper: Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection**

This is my own MemAE implementation, referring to the official source code and rewriting the model evaluation method myself. There is only one template for model training, so let's train it ourselves. I will update the training results in the future

## Requirements

- Python  >=3.6
- PyTorch >=1.0

**This code has no special requirements for the environment**



## **Train**

Train.py

**You can set your training parameters in args and run Python Train.py directly**

```python
class args():
    # Model setting
    MemDim = 2000
    EntropyLossWeight = 0.0002
    ShrinkThres = 0.0025
    checkpoint = r'memae-master/results/1.pth'

    # Dataset setting
    channels = 1
    size = 256
    videos_dir = r'memae-master\datasets\Test\Test001'
    time_steps = 16

    # For GPU training
    gpu = 0  # None
```



# Test

The test dataset needs to be manually created, and one video folder can be detected at a time. You need to place label files in the folder, with label file format: {‘imgpath’ ，  0/1} 0 representing normal

like this:

```txt
./UCSD_Anomaly_Dataset/UCSDped1/Test/Test001/014.tif 0
```

**Finally, the test will return the average AUC value of the test video**

