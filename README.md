#### Speech emotion recognition

> Bidirectional LSTM network with attention pooling for speech emotion recognition
>
> Environment:
>
> * Python 3.5

***

#### How to Install

```(bash)
pip install -r requirements.txt
```

***

#### Main requirements

> * librosa 0.6.2
> * Keras 2.2.4
> * Keras-Applications 1.0.6
> * Keras-Preprocessing 1.0.5
> * tensorflow-gpu 1.12.0
> * tensorflow 1.12.0
> * numpy 1.16.0
> * scipy 1.2.0
> * pyvad 0.0.8

------

#### Data

> https://www.kaggle.com/c/qia-hackathon2020

------

#### Preprocessing

> Preprocessing is composed of 5 steps.
>
> * load wavform
>
>   * sampling rate = 16kHz
>
> * denoising
>
>   * wiener filter
>
> * pre-emphasis filter
>
>   * coefficient = 0.97
>
> * voice activity detection 
>
>   * pyvad
>
> * log-scaled mel-spectrogram
>
>   * window size = 25ms
>
>   * stride = 10ms
>
>   * the number of mel bins = 40
>
>   * log scale
>     $$
>     mel_{log} = log(1 + w * mel), w = 1e+6
>     $$
>
> * zero padding
>
>   * maximum frame length = 400 (4 sec)

------

#### Network Architecture

> ![](.\img\network_architecture.png)
>
> *Network architecture*
>
> * Input
>
>   * shape = (max frame length, # of mel bins, 1)
>   * max frame length = 400
>   * the number of mel bins  = 40
>
> * conv2d
>
>   * kernel size = (5, 5)
>   * batchnorm
>   * relu activation
>
> * bi-LSTM
>
>   * bi-directional LSTM
>   * LSTM unit = 32
>
> * attention pooling
>
>   * input 
>
>
>   $$
>   x_{f,u}$
>   $$
>
>
>     * shape = (frame length with mpool (*f*), lstm unit(*u*))
>
>
>   * averagepooling1d for lstm unit direction
>
>     * 
>
>   * attention
>
>   * 
>
>     * $$
>       𝛼_𝑖=exp⁡(𝑓(𝑥)) / (∑8_𝑗exp⁡(𝑓(𝑥)) )
>       $$
>
>   * 
>
>     * $$
>       𝑓(𝑥)=𝑊^𝑇 𝑥
>       $$
>
>       where *W* is a trainable parameter from dense layer
>
>   * attention output
>
>     * $$
>       𝑎𝑡𝑡entive_x= ∑_𝑖𝛼_𝑖  𝑥_𝑖
>       $$
>
>       
>
> * dense
>   
>   * dropout 0.5

***

#### Training 

> * learning algorithm
>   * adam optimizer
>     * learning rate = 0.001
>     * beta_1 = 0.9
>     * beta_2 = 0.999
> * training loss
>   * cross entropy loss

***

#### Experimental result

> * validation accuracy
>   * 39.84 %

***

#### References

> * Li, Pengcheng, et al. "An attention pooling based representation learning method for speech emotion recognition." (2018).
