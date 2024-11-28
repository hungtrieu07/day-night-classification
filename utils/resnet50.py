from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, ZeroPadding2D, Add, GlobalAveragePooling2D, Dense
from keras.models import Model


class ResNet50:
    @staticmethod
    def identity_block(X, filters, kernel_size, stage, block):
        """
        Implementation of the identity block for ResNet50.

        Args:
        X: input tensor
        filters: list of integers, defining the number of filters in the CONV layers
        kernel_size: integer, specifying the shape of the middle CONV window
        stage: integer, used to name the layers
        block: string/character, used to name the layers
        
        Returns:
        Output tensor of the block
        """
        # Define base name for layers
        conv_name_base = f"res{stage}{block}_branch"
        bn_name_base = f"bn{stage}{block}_branch"

        # Retrieve filters
        F1, F2, F3 = filters

        # Save the input value for adding later
        X_shortcut = X

        # First component
        X = Conv2D(F1, (1, 1), strides=(1, 1), padding="valid", name=f"{conv_name_base}2a")(X)
        X = BatchNormalization(axis=3, name=f"{bn_name_base}2a")(X)
        X = Activation("relu")(X)

        # Second component
        X = Conv2D(F2, kernel_size, strides=(1, 1), padding="same", name=f"{conv_name_base}2b")(X)
        X = BatchNormalization(axis=3, name=f"{bn_name_base}2b")(X)
        X = Activation("relu")(X)

        # Third component
        X = Conv2D(F3, (1, 1), strides=(1, 1), padding="valid", name=f"{conv_name_base}2c")(X)
        X = BatchNormalization(axis=3, name=f"{bn_name_base}2c")(X)

        # Add shortcut
        X = Add()([X, X_shortcut])
        X = Activation("relu")(X)

        return X

    @staticmethod
    def convolutional_block(X, filters, kernel_size, stage, block, strides=(2, 2)):
        """
        Implementation of the convolutional block for ResNet50.

        Args:
        X: input tensor
        filters: list of integers, defining the number of filters in the CONV layers
        kernel_size: integer, specifying the shape of the middle CONV window
        stage: integer, used to name the layers
        block: string/character, used to name the layers
        strides: Tuple of integers, defining the strides
        
        Returns:
        Output tensor of the block
        """
        # Define base name for layers
        conv_name_base = f"res{stage}{block}_branch"
        bn_name_base = f"bn{stage}{block}_branch"

        # Retrieve filters
        F1, F2, F3 = filters

        # Save the input value for adding later
        X_shortcut = X

        # First component
        X = Conv2D(F1, (1, 1), strides=strides, padding="valid", name=f"{conv_name_base}2a")(X)
        X = BatchNormalization(axis=3, name=f"{bn_name_base}2a")(X)
        X = Activation("relu")(X)

        # Second component
        X = Conv2D(F2, kernel_size, strides=(1, 1), padding="same", name=f"{conv_name_base}2b")(X)
        X = BatchNormalization(axis=3, name=f"{bn_name_base}2b")(X)
        X = Activation("relu")(X)

        # Third component
        X = Conv2D(F3, (1, 1), strides=(1, 1), padding="valid", name=f"{conv_name_base}2c")(X)
        X = BatchNormalization(axis=3, name=f"{bn_name_base}2c")(X)

        # Shortcut path
        X_shortcut = Conv2D(F3, (1, 1), strides=strides, padding="valid", name=f"{conv_name_base}1")(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=f"{bn_name_base}1")(X_shortcut)

        # Add shortcut
        X = Add()([X, X_shortcut])
        X = Activation("relu")(X)

        return X

    @staticmethod
    def build(width, height, depth, classes):
        """
        Implementation of the ResNet50 architecture.

        Args:
        width: integer, input image width
        height: integer, input image height
        depth: integer, number of channels
        classes: integer, number of output classes
        
        Returns:
        A Model instance in Keras
        """
        input_shape = (height, width, depth)
        X_input = Input(input_shape)

        # Stage 1
        X = ZeroPadding2D((3, 3))(X_input)
        X = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(X)
        X = BatchNormalization(axis=3, name="bn_conv1")(X)
        X = Activation("relu")(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = ResNet50.convolutional_block(X, filters=[64, 64, 256], kernel_size=3, stage=2, block='a', strides=(1, 1))
        X = ResNet50.identity_block(X, filters=[64, 64, 256], kernel_size=3, stage=2, block='b')
        X = ResNet50.identity_block(X, filters=[64, 64, 256], kernel_size=3, stage=2, block='c')

        # Stage 3
        X = ResNet50.convolutional_block(X, filters=[128, 128, 512], kernel_size=3, stage=3, block='a')
        X = ResNet50.identity_block(X, filters=[128, 128, 512], kernel_size=3, stage=3, block='b')
        X = ResNet50.identity_block(X, filters=[128, 128, 512], kernel_size=3, stage=3, block='c')
        X = ResNet50.identity_block(X, filters=[128, 128, 512], kernel_size=3, stage=3, block='d')

        # Stage 4
        X = ResNet50.convolutional_block(X, filters=[256, 256, 1024], kernel_size=3, stage=4, block='a')
        X = ResNet50.identity_block(X, filters=[256, 256, 1024], kernel_size=3, stage=4, block='b')
        X = ResNet50.identity_block(X, filters=[256, 256, 1024], kernel_size=3, stage=4, block='c')
        X = ResNet50.identity_block(X, filters=[256, 256, 1024], kernel_size=3, stage=4, block='d')
        X = ResNet50.identity_block(X, filters=[256, 256, 1024], kernel_size=3, stage=4, block='e')
        X = ResNet50.identity_block(X, filters=[256, 256, 1024], kernel_size=3, stage=4, block='f')

        # Stage 5
        X = ResNet50.convolutional_block(X, filters=[512, 512, 2048], kernel_size=3, stage=5, block='a')
        X = ResNet50.identity_block(X, filters=[512, 512, 2048], kernel_size=3, stage=5, block='b')
        X = ResNet50.identity_block(X, filters=[512, 512, 2048], kernel_size=3, stage=5, block='c')

        # Average pooling
        X = GlobalAveragePooling2D()(X)

        # Output layer
        X = Dense(classes, activation="softmax", name="fc" + str(classes))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name="ResNet50")

        return model
