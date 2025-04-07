import torch
from PIL import Image


class ImageTransformer:
    """
    该类接受三维数组的图像作为输入，用于将图像转换为处理后的张量，以便后续处理。
    它可以与从监控抽取帧的ImageLoader组合使用。
    """

    def __init__(self, target_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        初始化图像变换器。

        :param target_size: 目标图像大小，默认为(224, 224)
        :param mean: 标准化均值，默认为ImageNet的均值
        :param std: 标准化标准差，默认为ImageNet的标准差
        """
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def _scale(self, image):
        """
        调整图像大小并标准化。

        :param image: 输入图像，类型为PIL图像
        :return: 放缩后的张量
        """
        # 调整图像大小
        resized_img = image.resize(self.target_size, resample=Image.Resampling.BILINEAR)
        resized = torch.asarray(resized_img)

        return resized

    def _gray_scaling(self, image):
        """
        将图像转换为灰度图像。

        :param image: 输入图像张量，形状为[C, H, W]
        :return: 灰度图像张量，形状为[1, H, W]
        """
        r, g, b = image[0], image[1], image[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.unsqueeze(0)

    def _enhance(self, image):
        """
        使用拉普拉斯算子对图像进行锐化。

        :param image: 输入图像张量，形状为[C, H, W]
        :return: 锐化后的图像张量
        """
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        enhanced = image + torch.abs(
            torch.nn.functional.conv2d(image.unsqueeze(0), laplacian_kernel, padding=1)
        )
        return enhanced

    def _normalize(self, image):
        """
        对图像进行Min - Max归一化。

        :param image: 输入图像张量
        :return: 归一化后的图像张量
        """
        min_val = torch.min(image)
        max_val = torch.max(image)
        denominator = max_val - min_val
        if denominator == 0:
            return image

        return (image - min_val) / denominator

    def transform(self, image):
        """
        对图像进行完整变换流程。

        :param image: 输入图像，可以是numpy数组或PIL图像
        :return: 处理后的图像张量
        """
        # 调整大小并标准化
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = self._scale(image)
        image = torch.as_tensor(image).permute(2, 0, 1).float()

        # 转换为灰度图像
        image = self._gray_scaling(image)
        # 锐化图像
        image = self._enhance(image)
        # 归一化
        image_tensor = self._normalize(image)

        return image_tensor