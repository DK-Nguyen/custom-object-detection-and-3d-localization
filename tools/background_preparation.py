#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import argparse
from pathlib import Path
import pathlib
import cv2
import os
from typing import List
import time


class BackgroundPreparation:
    """
    Read the background, resize it if the size is not suitable,
    put the foreground (the tank) in the middle bottom of the background)"""
    def __init__(self,
                 args):
        """
        :param args: the argument parser
        :type args: parser.parse_args()
        """
        self.input_dir: pathlib.PosixPath = Path(args.input_dir)
        self.output_dir: pathlib.PosixPath = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_output_types: list = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types: list = ['.png', '.jpg', '.jpeg']
        self.output_type: str = f'.{args.output_type}'

    def _validate_argument_parser(self, args):
        """
        Validate the width and height
        :param args: parser.parse_args()
        :return:
        """
        assert args.width >= 64, 'width must be greater than 64'
        self.width = args.width
        assert args.height >= 64, 'height must be greater than 64'
        self.height = args.height

        assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}'

    def compose_images(self,
                       height: int,
                       width: int,
                       position: str = 'middle_bottom') -> None:
        """
        Read the background images, resize them to the (height, width) dimension,
        put the foreground on the desired position

        :param height: the height of the output image
        :type height: int
        :param width: the width of the ouput image
        :type width: int
        :param position: the relative position to put the foreground on the background
        :return: none
        """
        # Open background and convert to RGBA
        background_path = self.input_dir.joinpath('backgrounds')
        bg_im_list = sorted(background_path.glob('*'))
        foreground_path = self.input_dir.joinpath('foregrounds')
        fg_im_list = sorted(foreground_path.glob('*'))

        i = 0
        for bg_im_path in bg_im_list:
            for fg_im_path in fg_im_list:
                fg_im = Image.open(fg_im_path).convert('RGBA')
                bg_im = Image.open(bg_im_path).convert('RGBA')
                bg_im = bg_im.resize((height, width))
                # put the foreground to the desired position
                if position == 'middle_bottom':
                    x_pos = int((width-fg_im.size[1])/2)
                    y_pos = int(height-fg_im.size[0])
                else:
                    x_pos = 0
                    y_pos = 0
                bg_im.paste(fg_im, (x_pos, y_pos), mask=fg_im)
                save_filename = f'{i:0{4}}{self.output_type}'  # e.g. 00000023.jpg
                output_path = Path(self.output_dir / save_filename)
                bg_im.convert('RGB').save(output_path)
                i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescaling Images")
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=False,
                        help="the input directory that contains images to rescale", default="./input")
    parser.add_argument("--width", type=int, dest="width", required=False,
                        help="output image pixel width", default=512)
    parser.add_argument("--height", type=int, dest="height", required=False,
                        help="output image pixel height", default=512)
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=False,
                        help="the input directory that contains images to rescale", default="./output")
    parser.add_argument("--output_type", type=str, dest="output_type",
                        help="png or jpg (default)", default='jpg')

    args = parser.parse_args()

    bg_object = BackgroundPreparation(args)
    bg_object.compose_images(args.height, args.width)
