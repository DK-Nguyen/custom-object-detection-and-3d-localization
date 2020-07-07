
from PIL import Image
from pathlib import Path
import logging
from tqdm.auto import tqdm

__all__ = ["HardNegativeBackgroundPreparation"]

# A logger for this file
log = logging.getLogger(__name__)


class HardNegativeBackgroundPreparation:
    """
    Prepare the background images for the dataset (which contain true negative
    foregrounds to help the training process).
    """
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 output_width: int = 512,
                 output_height: int = 512,
                 output_type: str = 'png'
                 ) -> None:
        """
        :param input_dir: the input directory that contains the foregrounds and backgrounds
        :type input_dir: str
        :param output_dir: the output directory that contains output images
        :type output_dir: str
        :param output_width: output image width in pixels
        :type output_width: int (default: 512)
        :param output_height: output image height in pixels
        :type output_height: int (default: 512)
        :param output_type: output image type (jpg or png)
        :type output_height: str (default: 512)
        """
        # log.info("Preparing true-negative images")
        self.input_dir: Path = Path(input_dir)
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_output_types: list = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types: list = ['.png', '.jpg', '.jpeg']
        self.output_type: str = f'.{output_type}'
        self.width = output_width
        self.height = output_height

    def _validate_arguments(self) -> None:
        """
        Validate the width, height, and output types
        """
        assert self.width >= 64, 'width must be greater than 64'
        assert self.height >= 64, 'height must be greater than 64'
        assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}'

    def compose_images(self,
                       position: str = 'middle_bottom') -> None:
        """
        Read the background images, resize them to the given (height, width),
        put the foreground on the desired position
        (currently there is one position: middle_bottom, otherwise the foreground will be at (0,0))

        :param height: the height of the output image
        :type height: int
        :param width: the width of the ouput image
        :type width: int
        :param position: the relative position to put the foreground on the background
        :return: none
        """
        self._validate_arguments()
        # Open background and convert to RGBA
        background_path = self.input_dir.absolute().joinpath('backgrounds')
        bg_im_list = sorted(background_path.glob('*'))
        foreground_path = self.input_dir.absolute().joinpath('foregrounds')
        fg_im_list = sorted(foreground_path.glob('*'))

        im_counter = 0
        for bg_im_path in tqdm(bg_im_list):
            for fg_im_path in fg_im_list:
                fg_im = Image.open(fg_im_path).convert('RGBA')
                bg_im = Image.open(bg_im_path).convert('RGBA')
                bg_im = bg_im.resize((self.height, self.width))
                # put the foreground to the desired position
                if position == 'middle_bottom':
                    x_pos = int((self.width - fg_im.size[1]) / 2)
                    y_pos = int(self.height - fg_im.size[0])
                else:
                    x_pos = 0
                    y_pos = 0
                bg_im.paste(fg_im, (x_pos, y_pos), mask=fg_im)
                save_filename = f'{im_counter:0{4}}{self.output_type}'  # e.g. 0001.png
                output_path = Path(self.output_dir / save_filename)
                bg_im.convert('RGB').save(output_path)
                im_counter += 1

        log.info(f'Done preparing {im_counter} hard negative background images')
