



|Layer (type)           |      Output Shape        |      Param #  | 
|:--------------------:|:--------------------:|:-------------:|
|lambda_18 (Lambda)       |    (None, 160, 320, 3)   |    0         |
|cropping2d_18 (Cropping2D) |  (None, 65, 320, 3)    |    0         
|conv2d_49 (Conv2D)        | (None, 31, 158, 24)    |   1824      
|conv2d_50 (Conv2D)        | (None, 14, 77, 36)    |    21636     
|conv2d_51 (Conv2D)        | (None, 5, 37, 48)     |    43248     
|conv2d_52 (Conv2D)        |  (None, 3, 35, 64)    |     27712     
|conv2d_53 (Conv2D)        |   (None, 1, 33, 70)    |     40390     
|flatten_12 (Flatten)      |    (None, 2310)        |      0         
|dense_37 (Dense)          |     (None, 100)        |       231100    
|dense_38 (Dense)         |      (None, 50)       |         5050      
|dense_39 (Dense)        |       (None, 10)         |       510       
|dense_40 (Dense)       |        (None, 1)          |       11        


=================================================================
Total params: 371,481

Trainable params: 371,481

Non-trainable params: 0

_________________________________________________________________
None

