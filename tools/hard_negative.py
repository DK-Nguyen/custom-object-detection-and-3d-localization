
from random import randint
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
                 input_dir: Path,
                 output_dir: Path,
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
        self.input_dir: Path = input_dir
        self.background_dir = self.input_dir.absolute().joinpath('backgrounds')
        self.foreground_dir = self.input_dir.absolute().joinpath('foregrounds')
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_output_types: list = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types: list = ['.png', '.jpg', '.jpeg']
        self.output_type: str = f'.{output_type}'
        self.width = output_width
        self.height = output_height

    def _validate_arguments(self) -> None:
        """
        Validate the directories, width, height, and output types
        """
        assert self.input_dir.exists(), f'the {self.input_dir} does not exist'
        assert self.background_dir.exists(), f'the {self.background_dir} does not exist'
        assert self.foreground_dir.exists(), f'the {self.foreground_dir} does not exist'
        assert self.width >= 64, 'width must be greater than 64'
        assert self.height >= 64, 'height must be greater than 64'
        assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}'

    def compose_images(self,
                       position: str = 'middle_bottom') -> None:
        """
        Read the background images, resize them to the given (height, width),
        put the foreground on the desired position
        (currently there is one position: middle_bottom, otherwise the foreground will
        put at a random place on the background image)

        :param position: the relative position to put the foreground on the background
        :type position: string. Default: 'middle_bottom'
        :return: none
        """
        self._validate_arguments()
        # Open background and convert to RGBA
        bg_im_list = sorted(self.background_dir.glob('*'))
        fg_im_list = sorted(self.foreground_dir.glob('*'))

        im_counter = 0
        for bg_im_path in tqdm(bg_im_list):
            for fg_im_path in fg_im_list:
                fg_im = Image.open(fg_im_path).convert('RGBA')
                bg_im = Image.open(bg_im_path).convert('RGBA')
                bg_width, bg_height = bg_im.size
                fg_width, fg_height = fg_im.size
                if bg_width < fg_width*2:
                    raise Exception(f"Background image {bg_im_path}'s width is less than"
                                    f"2 times foreground image {fg_im_path}'s width")
                if bg_height < fg_height*2:
                    raise Exception(f"Background image {bg_im_path}'s height is less than"
                                    f"foreground image {fg_im_path}'s height + 100")
                # put the foreground to the desired position
                if position == 'middle_bottom':
                    x_pos = int((bg_width - fg_width) / 2)
                    y_pos = int(bg_height - fg_im.height)
                else:
                    x_pos = randint(fg_width, bg_width-fg_width)
                    y_pos = randint(fg_height, bg_height-fg_height)
                bg_im.paste(fg_im, (x_pos, y_pos), mask=fg_im)
                bg_im = bg_im.resize((self.width, self.height))
                save_filename = f'{im_counter:0{4}}{self.output_type}'  # e.g. 0000.png
                output_path = Path(self.output_dir / save_filename)
                bg_im.convert('RGB').save(output_path)
                im_counter += 1

        log.info(f'Done preparing {im_counter} hard negative background images, '
                 f'output directory: {self.output_dir}')

