# Image Captioning
Image Captioning model using InceptionV3 and Transformer model on MS-COCO Dataset

## Dataset
- The dataset used is MS-COCO Captions Dataset 
- [Link to Dataset](https://cocodataset.org/#captions-2015)

## Technologies Used
![Python](https://img.shields.io/badge/-Python-FFFFFF?style=flat&logo=python&logoColor=3776AB)&nbsp;&nbsp;&nbsp;
![Numpy](https://img.shields.io/badge/-NumPy-FFFFFF?style=flat&logo=numpy&logoColor=013243)&nbsp;&nbsp;&nbsp;
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FFFFFF?style=flat&logo=tensorflow&logoColor=FF6F00)&nbsp;&nbsp;&nbsp;
![Matplotlib](https://img.shields.io/badge/-Matplotlib-FFFFFF?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAEk0lEQVQ4T32VfUgbZxzH73KX95iX80xitMbK5qTWdBoMU7p28a20sCG2f7SwIYylY2+Q1o51yrSuKoOKupWxabZANxgbSPWPdSNCfKmCGOusZhYtlNoaF/N6iUkuL5fcjSerLrO6++fgud/zud/3nu/vezC0/4VUVFScbGpq0ut0uiMymUwBygiCcNlstgcjIyPjCwsLdyEISu3dDu9d0Gg0r/b39195+SX1i8G5UUzdeGkzFoshoE4gECTBnaZpeGpqymE0GnuXl5enMxmZQJbBYPig69pnBvrukFK6dkcIU3GW4/yPm2wxToFNPB4vxWKxIAzD4iwWiyFJEjIajbdMJtPX4D2gZhdoMBg+6ujoeA+TSRPEt81FcmIFRRGEcddf84jKzwT+AXJSoVCQQ1FMeh8AO51Ofmdn5zcmk+nmLhDInJ2d7efxeLDdbpcqROxk1s/v5AkSQdZGQX0Mqr/sAsVhaoMXxU1yZfwTBy4pIldXV8VlZWWBWCzGVFVVXQLywZsQq9V6u6amJs/r9fKysrIon8/HDf85IS6c7sFCAhXN+fD2IwRBmPXwT4owsyTKS17dgGEYys7OjodCITaO47Hx8fHN2traJlir1eptNlsv6MDv93PBQwAGhZxZs0yxNipae2PQJczOjTu5N/IPIe86k6QkJRKJkju1QDrYr9PprsBdXV3X29raTm86ngixbHkcRVF6e3ubLZPJEsv3F7Hc8U48ojkbTBUfjRPMpEQUPR0oKSkJEgTBEYvFFIqiDFAE4D09Pb/BFotluKGhoXD13i1VLvydhEhqEvycUwFYWEGibAG9uXpfzNh/lfgrjlEqSWUAx5UxoVCY5HK5KaAEgNxuN08ul8fGxsbWYZvNNlNaWip6uPhDrkZ8MyuZTMEoijCRGMKKQGXxJP94ZHJyS+qipDAjeoU8UZkTUGKcBI7j8R2gx+Ph5eTkxFZWVsL/CwwxR+N/PFYjWyEckgiwxHH9qS2JRJrwer1A4v7APZLFgaSG4uINwblHhei9NUhQUw6HZGo6RZIhVDzzu0jyWrNHnl9I7ifZYrGsw93d3Z+3traeyTwUv5/gWuZISWkeQR0+rA7bE3YpOMUCJ0Kx7lxXii7ceArL8ilgsecO5SDbhMNhtKCgIEJRFGv4r2E1AJ5TnXvim/kF5099iftOfux7ofb8FviO/7FNprF9Ph8PBACwBIfDoWEYZoB5h1xDxQB4UXHxodfr5kWG2/PljmmuV/t2kFv9pk+uUEStVqujrq7ubHom94yerLi4eJvP5ydBdwC+GFrEQF15Vrkf+DMR2UYTgxeKuIF1tufYW0H89ctPq6urd0cvnT4gHNrb299XqVQkQRBchmHSowW6JEkS3YkvhmFgYOTQ42WhcKoPD9a3u7/4anDAbDb/Gw7P8iwdXwMDA80CgSCdeWAUaZqGotFoGgi6zowvj9uFXv209Xuz2fx8fO2EJJDf19fXotfrD4HMA+uZHe4E7MTExEZLS0vv0tLSzEEBm7mOaLXaE42NjfrKysojGIYpn4XH1vz8/IPR0dGJg34BfwP5MXT+u6N2TgAAAABJRU5ErkJggg==)&nbsp;&nbsp;&nbsp;
![Colaboratory](https://img.shields.io/badge/-Google%20Colab-FFFFFF?style=flat&logo=google-colab&logoColor=F9AB00)&nbsp;&nbsp;&nbsp;

## Installations Required
For running the source code on your local machine, the following dependencies are required.
- Python
- Numpy
- TensorFlow
- Matplotlib
- Pillow
- NLTK

## Launch
For local installation, follow these steps:
1. Download source code from this repository or click [here](https://github.com/rishabh1323/Image-Captioning/archive/refs/heads/main.zip).
2. Extract files to desired directory.
3. [Download](https://www.python.org/downloads/) and install Python3 if not done already.
4. Create a new python virtual environment.
```
python3 -m venv tutorial-env
```
5. Once you’ve created a virtual environment, you may activate it.  

`On Windows, run:`
```
tutorial-env\Scripts\activate.bat
```
`On Unix or MacOS, run:`
```
source tutorial-env/bin/activate
```
> Refer to [python documentation](https://docs.python.org/3/tutorial/venv.html) for more information on virtual environments.  
6. Install the required dependecies.
```
pip install numpy tensorflow matplotlib pillow nltk
```
7. Launch and run the python file now.
