# nhập các gói cần thiết
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

class ResNet:
	@staticmethod
	def residual_module(data, K, stride, chanDim, red=False,
		reg=0.0001, bnEps=2e-5, bnMom=0.9):
		# nhánh tắt của mô-đun ResNet phải là
                # khởi tạo làm dữ liệu đầu vào (danh tính)
		shortcut = data

		# khối đầu tiên của mô-đun ResNet là CONV 1x1
		bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(data)
		act1 = Activation("relu")(bn1)
		conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
			kernel_regularizer=l2(reg))(act1)

		# khối thứ hai của mô-đun ResNet là CONV 3x3
		bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(conv1)
		act2 = Activation("relu")(bn2)
		conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
			padding="same", use_bias=False,
			kernel_regularizer=l2(reg))(act2)

		# khối thứ ba của mô-đun ResNet là một tập hợp 1x1 khác
                # CONV
		bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(conv2)
		act3 = Activation("relu")(bn3)
		conv3 = Conv2D(K, (1, 1), use_bias=False,
			kernel_regularizer=l2(reg))(act3)

		# nếu chúng ta muốn giảm kích thước không gian, hãy áp dụng lớp CONV cho
                # lối tắt
		if red:
			shortcut = Conv2D(K, (1, 1), strides=stride,
				use_bias=False, kernel_regularizer=l2(reg))(act1)

		# thêm phím tắt và CONV cuối cùng
		x = add([conv3, shortcut])

		# trả về phần bổ sung làm đầu ra của mô-đun ResNet
		return x

	@staticmethod
	def build(width, height, depth, classes, stages, filters,
		reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
		# khởi tạo hình dạng đầu vào là "kênh cuối cùng" và
                # thứ nguyên kênh
		inputShape = (height, width, depth)
		chanDim = -1

		# nếu chúng ta đang sử dụng "kênh trước", hãy cập nhật hình dạng đầu vào
                # và thứ nguyên kênh
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		#đặt đầu vào và sau đó áp dụng BN theo sau là CONV
		inputs = Input(shape=inputShape)
		x = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(inputs)
		x = Conv2D(filters[0], (3, 3), use_bias=False,
			padding="same", kernel_regularizer=l2(reg))(x)

		# lặp qua số giai đoạn
		for i in range(0, len(stages)):
			# khởi tạo sải bước, sau đó áp dụng mô-đun dư
                        # được sử dụng để giảm kích thước không gian của âm lượng đầu vào
			stride = (1, 1) if i == 0 else (2, 2)
			x = ResNet.residual_module(x, filters[i + 1], stride,
				chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

			# lặp lại số lớp trong giai đoạn
			for j in range(0, stages[i] - 1):
				# áp dụng mô-đun ResNet
				x = ResNet.residual_module(x, filters[i + 1],
					(1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

		# apply BN => ACT => POOL
		x = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(x)
		x = Activation("relu")(x)
		x = AveragePooling2D((7, 7))(x)

		# softmax classifier
		x = Flatten()(x)
		x = Dense(classes, kernel_regularizer=l2(reg))(x) 
		x = Activation("softmax")(x)

		# create the model
		model = Model(inputs, x, name="resnet")

		# trả về kiến ​​trúc mạng đã được xây dựng
		return model
