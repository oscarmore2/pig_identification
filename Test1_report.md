#Define Hyper-paramenters
display_step = 1
learning_rate = 0.0001
batch_size = 256
epochs = 256
num_epochs = 50
bast_val = 0.0
last_improvement = 0
num_residual_blocks = 4
train_ema_decay = 0.95
require_improvement = 1000
keep_probability = 0.5

best_validation_accuracy = 0.0

H, W, C = 32, 32, 3
image_shape = (H, W, C)
image_flat_shape = H * W * C
num_classes = 10


Epoch  1, CIFAR-10 Batch 1:  1511853015.8144617: loss = 2.0782 (14.3 examples/sec; 17.960 sec/batch)
Loss:     2.0026 Validation Accuracy: 0.288400
----------------------------
Epoch  2, CIFAR-10 Batch 1:  1511853036.3611429: loss = 1.8490 (15.9 examples/sec; 16.149 sec/batch)
Loss:     1.7290 Validation Accuracy: 0.334600
----------------------------
Epoch  3, CIFAR-10 Batch 1:  1511853055.6675456: loss = 1.6854 (15.8 examples/sec; 16.173 sec/batch)
Loss:     1.5319 Validation Accuracy: 0.397000
----------------------------
Epoch  4, CIFAR-10 Batch 1:  1511853074.979305: loss = 1.5729 (15.8 examples/sec; 16.166 sec/batch)
Loss:     1.4076 Validation Accuracy: 0.418000
----------------------------
Epoch  5, CIFAR-10 Batch 1:  1511853094.2872472: loss = 1.3555 (15.8 examples/sec; 16.166 sec/batch)
Loss:     1.2113 Validation Accuracy: 0.461000
----------------------------
Epoch  6, CIFAR-10 Batch 1:  1511853113.9766233: loss = 1.3337 (15.8 examples/sec; 16.171 sec/batch)
Loss:     1.1749 Validation Accuracy: 0.492200
----------------------------
Epoch  7, CIFAR-10 Batch 1:  1511853133.4965842: loss = 1.0889 (15.7 examples/sec; 16.289 sec/batch)
Loss:     0.9543 Validation Accuracy: 0.515600
----------------------------
Epoch  8, CIFAR-10 Batch 1:  1511853152.8827474: loss = 1.0034 (15.8 examples/sec; 16.238 sec/batch)
Loss:     0.8291 Validation Accuracy: 0.543400
----------------------------
Epoch  9, CIFAR-10 Batch 1:  1511853172.2003834: loss = 1.0045 (15.8 examples/sec; 16.175 sec/batch)
Loss:     0.8135 Validation Accuracy: 0.537600
----------------------------
Epoch 10, CIFAR-10 Batch 1:  1511853191.5666008: loss = 0.8890 (15.8 examples/sec; 16.211 sec/batch)
Loss:     0.7435 Validation Accuracy: 0.551400
----------------------------
Epoch 11, CIFAR-10 Batch 1:  1511853210.9859655: loss = 0.6843 (15.7 examples/sec; 16.281 sec/batch)
Loss:     0.5695 Validation Accuracy: 0.568400
----------------------------
Epoch 12, CIFAR-10 Batch 1:  1511853230.3120234: loss = 0.6496 (15.9 examples/sec; 16.137 sec/batch)
Loss:     0.5088 Validation Accuracy: 0.568000
----------------------------
Epoch 13, CIFAR-10 Batch 1:  1511853249.8359616: loss = 0.6227 (15.7 examples/sec; 16.273 sec/batch)
Loss:     0.4899 Validation Accuracy: 0.577000
----------------------------
Epoch 14, CIFAR-10 Batch 1:  1511853269.2708247: loss = 0.5027 (15.8 examples/sec; 16.216 sec/batch)
Loss:     0.4380 Validation Accuracy: 0.581200
----------------------------
Epoch 15, CIFAR-10 Batch 1:  1511853288.9952998: loss = 0.4621 (15.8 examples/sec; 16.177 sec/batch)
Loss:     0.3500 Validation Accuracy: 0.573800
----------------------------
Epoch 16, CIFAR-10 Batch 1:  1511853308.4246068: loss = 0.4181 (15.7 examples/sec; 16.258 sec/batch)
Loss:     0.3332 Validation Accuracy: 0.587000
----------------------------
Epoch 17, CIFAR-10 Batch 1:  1511853327.737817: loss = 0.3769 (15.8 examples/sec; 16.175 sec/batch)
Loss:     0.2792 Validation Accuracy: 0.593200
----------------------------
Epoch 18, CIFAR-10 Batch 1:  1511853347.3290224: loss = 0.3513 (15.6 examples/sec; 16.406 sec/batch)
Loss:     0.2600 Validation Accuracy: 0.592200
----------------------------
Epoch 19, CIFAR-10 Batch 1:  1511853367.0438828: loss = 0.2784 (15.6 examples/sec; 16.437 sec/batch)
Loss:     0.2087 Validation Accuracy: 0.588200
----------------------------
Epoch 20, CIFAR-10 Batch 1:  1511853386.392053: loss = 0.3150 (15.8 examples/sec; 16.201 sec/batch)
Loss:     0.2197 Validation Accuracy: 0.605800
----------------------------
c:\users\administrator\anaconda3\envs\intro-to-rnns\lib\site-packages\matplotlib\pyplot.py:524: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
  max_open_warning, RuntimeWarning)
Epoch 21, CIFAR-10 Batch 1:  1511853405.6841564: loss = 0.1842 (15.8 examples/sec; 16.152 sec/batch)
Loss:     0.1541 Validation Accuracy: 0.590000
----------------------------
Epoch 22, CIFAR-10 Batch 1:  1511853425.0269246: loss = 0.1786 (15.8 examples/sec; 16.167 sec/batch)
Loss:     0.1271 Validation Accuracy: 0.591200
----------------------------
Epoch 23, CIFAR-10 Batch 1:  1511853444.4126527: loss = 0.1846 (15.8 examples/sec; 16.224 sec/batch)
Loss:     0.1185 Validation Accuracy: 0.593800
----------------------------
Epoch 24, CIFAR-10 Batch 1:  1511853464.0248804: loss = 0.1462 (15.6 examples/sec; 16.363 sec/batch)
Loss:     0.0937 Validation Accuracy: 0.604400
----------------------------
Epoch 25, CIFAR-10 Batch 1:  1511853484.1576264: loss = 0.1562 (15.6 examples/sec; 16.382 sec/batch)
Loss:     0.0986 Validation Accuracy: 0.607400
----------------------------
Epoch 26, CIFAR-10 Batch 1:  1511853503.860241: loss = 0.0893 (15.5 examples/sec; 16.549 sec/batch)
Loss:     0.0623 Validation Accuracy: 0.611000
----------------------------
Epoch 27, CIFAR-10 Batch 1:  1511853523.6669772: loss = 0.0661 (15.4 examples/sec; 16.631 sec/batch)
Loss:     0.0528 Validation Accuracy: 0.620200
----------------------------
Epoch 28, CIFAR-10 Batch 1:  1511853543.0480692: loss = 0.1723 (15.8 examples/sec; 16.229 sec/batch)
Loss:     0.0751 Validation Accuracy: 0.617800
----------------------------
Epoch 29, CIFAR-10 Batch 1:  1511853562.5106018: loss = 0.1397 (15.7 examples/sec; 16.318 sec/batch)
Loss:     0.0688 Validation Accuracy: 0.617200
----------------------------
Epoch 30, CIFAR-10 Batch 1:  1511853581.9827187: loss = 0.1067 (15.8 examples/sec; 16.222 sec/batch)
Loss:     0.0580 Validation Accuracy: 0.626200
----------------------------
Epoch 31, CIFAR-10 Batch 1:  1511853601.5220416: loss = 0.1491 (15.7 examples/sec; 16.343 sec/batch)
Loss:     0.0819 Validation Accuracy: 0.621200
----------------------------
Epoch 32, CIFAR-10 Batch 1:  1511853620.9171686: loss = 0.0397 (15.8 examples/sec; 16.209 sec/batch)
Loss:     0.0353 Validation Accuracy: 0.634600
----------------------------
Epoch 33, CIFAR-10 Batch 1:  1511853640.1799793: loss = 0.0540 (15.9 examples/sec; 16.133 sec/batch)
Loss:     0.0432 Validation Accuracy: 0.627800
----------------------------
Epoch 34, CIFAR-10 Batch 1:  1511853659.6067588: loss = 0.0460 (15.7 examples/sec; 16.266 sec/batch)
Loss:     0.0416 Validation Accuracy: 0.622400
----------------------------
Epoch 35, CIFAR-10 Batch 1:  1511853679.0867906: loss = 0.0589 (15.7 examples/sec; 16.321 sec/batch)
Loss:     0.0206 Validation Accuracy: 0.605600
----------------------------
Epoch 36, CIFAR-10 Batch 1:  1511853698.6638923: loss = 0.0380 (15.7 examples/sec; 16.337 sec/batch)
Loss:     0.0204 Validation Accuracy: 0.621800
----------------------------
Epoch 37, CIFAR-10 Batch 1:  1511853718.1266925: loss = 0.0582 (15.8 examples/sec; 16.220 sec/batch)
Loss:     0.0222 Validation Accuracy: 0.621000
----------------------------
Epoch 38, CIFAR-10 Batch 1:  1511853738.2103143: loss = 0.0350 (15.8 examples/sec; 16.240 sec/batch)
Loss:     0.0141 Validation Accuracy: 0.615600
----------------------------
Epoch 39, CIFAR-10 Batch 1:  1511853757.5400019: loss = 0.0253 (15.8 examples/sec; 16.168 sec/batch)
Loss:     0.0136 Validation Accuracy: 0.619000
----------------------------
Epoch 40, CIFAR-10 Batch 1:  1511853776.8767524: loss = 0.0259 (15.8 examples/sec; 16.192 sec/batch)
Loss:     0.0138 Validation Accuracy: 0.615200
----------------------------
Epoch 41, CIFAR-10 Batch 1:  1511853796.3431938: loss = 0.0518 (15.7 examples/sec; 16.268 sec/batch)
Loss:     0.0231 Validation Accuracy: 0.626600
----------------------------
Epoch 42, CIFAR-10 Batch 1:  1511853815.9789443: loss = 0.0384 (15.6 examples/sec; 16.414 sec/batch)
Loss:     0.0173 Validation Accuracy: 0.616600
----------------------------
Epoch 43, CIFAR-10 Batch 1:  1511853835.5104423: loss = 0.0193 (15.7 examples/sec; 16.265 sec/batch)
Loss:     0.0154 Validation Accuracy: 0.626800
----------------------------
Epoch 44, CIFAR-10 Batch 1:  1511853855.1011066: loss = 0.0329 (15.6 examples/sec; 16.426 sec/batch)
Loss:     0.0095 Validation Accuracy: 0.623200
----------------------------
Epoch 45, CIFAR-10 Batch 1:  1511853874.7085323: loss = 0.0122 (15.6 examples/sec; 16.431 sec/batch)
Loss:     0.0132 Validation Accuracy: 0.626400
----------------------------
Epoch 46, CIFAR-10 Batch 1:  1511853894.4029224: loss = 0.0221 (15.5 examples/sec; 16.487 sec/batch)
Loss:     0.0096 Validation Accuracy: 0.638000
----------------------------
Epoch 47, CIFAR-10 Batch 1:  1511853914.0124304: loss = 0.0459 (15.6 examples/sec; 16.426 sec/batch)
Loss:     0.0112 Validation Accuracy: 0.631800
----------------------------
Epoch 48, CIFAR-10 Batch 1:  1511853933.6013088: loss = 0.0215 (15.7 examples/sec; 16.325 sec/batch)
Loss:     0.0100 Validation Accuracy: 0.634800
----------------------------
Epoch 49, CIFAR-10 Batch 1:  1511853953.1760275: loss = 0.0367 (15.7 examples/sec; 16.335 sec/batch)
Loss:     0.0073 Validation Accuracy: 0.626400
----------------------------
Epoch 50, CIFAR-10 Batch 1:  1511853972.6419263: loss = 0.0269 (15.7 examples/sec; 16.311 sec/batch)
Loss:     0.0126 Validation Accuracy: 0.620400
----------------------------
Epoch 51, CIFAR-10 Batch 1:  1511853992.288117: loss = 0.0262 (15.6 examples/sec; 16.385 sec/batch)
Loss:     0.0053 Validation Accuracy: 0.625400
----------------------------
Epoch 52, CIFAR-10 Batch 1:  1511854012.003618: loss = 0.0135 (15.5 examples/sec; 16.518 sec/batch)
Loss:     0.0085 Validation Accuracy: 0.619000
----------------------------
Epoch 53, CIFAR-10 Batch 1:  1511854032.6121175: loss = 0.0095 (15.6 examples/sec; 16.363 sec/batch)
Loss:     0.0051 Validation Accuracy: 0.636400
----------------------------
Epoch 54, CIFAR-10 Batch 1:  1511854052.0861912: loss = 0.0234 (15.7 examples/sec; 16.288 sec/batch)
Loss:     0.0086 Validation Accuracy: 0.626800
----------------------------
Epoch 55, CIFAR-10 Batch 1:  1511854071.4019759: loss = 0.0066 (15.8 examples/sec; 16.182 sec/batch)
Loss:     0.0037 Validation Accuracy: 0.630400
----------------------------
Epoch 56, CIFAR-10 Batch 1:  1511854092.4399521: loss = 0.0043 (14.3 examples/sec; 17.865 sec/batch)
Loss:     0.0033 Validation Accuracy: 0.631200
----------------------------
Epoch 57, CIFAR-10 Batch 1:  1511854113.1924992: loss = 0.0641 (15.2 examples/sec; 16.865 sec/batch)
Loss:     0.0073 Validation Accuracy: 0.631400
----------------------------
Epoch 58, CIFAR-10 Batch 1:  1511854132.6971684: loss = 0.0606 (15.6 examples/sec; 16.384 sec/batch)
Loss:     0.0186 Validation Accuracy: 0.634600
----------------------------
Epoch 59, CIFAR-10 Batch 1:  1511854152.2252522: loss = 0.0218 (15.7 examples/sec; 16.354 sec/batch)
Loss:     0.0101 Validation Accuracy: 0.624200
----------------------------
Epoch 60, CIFAR-10 Batch 1:  1511854171.6529768: loss = 0.0077 (15.8 examples/sec; 16.245 sec/batch)
Loss:     0.0064 Validation Accuracy: 0.634600
----------------------------
Epoch 61, CIFAR-10 Batch 1:  1511854200.6624095: loss = 0.0054 (10.1 examples/sec; 25.436 sec/batch)
Loss:     0.0031 Validation Accuracy: 0.638000
----------------------------
Epoch 62, CIFAR-10 Batch 1:  1511854220.1257148: loss = 0.0627 (15.9 examples/sec; 16.147 sec/batch)
Loss:     0.0057 Validation Accuracy: 0.641600
----------------------------
Epoch 63, CIFAR-10 Batch 1:  1511854239.7453673: loss = 0.0061 (15.6 examples/sec; 16.419 sec/batch)
Loss:     0.0058 Validation Accuracy: 0.642800
----------------------------
Epoch 64, CIFAR-10 Batch 1:  1511854259.579084: loss = 0.0471 (15.9 examples/sec; 16.148 sec/batch)
Loss:     0.0039 Validation Accuracy: 0.632600
----------------------------
Epoch 65, CIFAR-10 Batch 1:  1511854279.2068586: loss = 0.0059 (15.7 examples/sec; 16.283 sec/batch)
Loss:     0.0051 Validation Accuracy: 0.642600
----------------------------
Epoch 66, CIFAR-10 Batch 1:  1511854298.8033066: loss = 0.0080 (15.6 examples/sec; 16.444 sec/batch)
Loss:     0.0027 Validation Accuracy: 0.645200
----------------------------
Epoch 67, CIFAR-10 Batch 1:  1511854318.2324564: loss = 0.0077 (15.7 examples/sec; 16.295 sec/batch)
Loss:     0.0059 Validation Accuracy: 0.638400
----------------------------
Epoch 68, CIFAR-10 Batch 1:  1511854337.756717: loss = 0.0018 (15.7 examples/sec; 16.352 sec/batch)
Loss:     0.0019 Validation Accuracy: 0.641000
----------------------------
Epoch 69, CIFAR-10 Batch 1:  1511854357.192339: loss = 0.0034 (15.8 examples/sec; 16.231 sec/batch)
Loss:     0.0018 Validation Accuracy: 0.639400
----------------------------