from ashlar import filepattern
from skimage.filters import gaussian
from skimage.util import invert
import skimage.exposure
import numpy as np

class FilePatternReaderRescale(filepattern.FilePatternReader):

    def __init__(self, path, pattern, overlap, pixel_size=1, do_rescale=False, WGAchannel = None, no_rescale_channel = "Alexa488", rescale_range = (1, 99)):
        super().__init__(path, pattern, overlap, pixel_size=pixel_size)
        self.do_rescale = do_rescale
        self.WGAchannel = WGAchannel
        self.no_rescale_channel = no_rescale_channel
        self.rescale_range = rescale_range

    @staticmethod
    def rescale(img, rescale_range = (1, 99), cutoff_threshold = None):
        img = skimage.util.img_as_float32(img)
        cutoff1, cutoff2 = rescale_range
        if cutoff_threshold is not None:
            if img.max() > (cutoff_threshold/65535):
                _img = img.copy()
                _img[_img > (cutoff_threshold/65535)] = 0
                p1 = np.percentile(_img, cutoff1)
                p99 = np.percentile(_img, cutoff2)
            else:
                p1 = np.percentile(img, cutoff1)
                p99 = np.percentile(img, cutoff2)
        else:
            p1 = np.percentile(img, cutoff1)
            p99 = np.percentile(img, cutoff2)
        img = skimage.exposure.rescale_intensity(img, 
                                                  in_range=(p1, p99), 
                                                  out_range=(0, 1))
        return((img * 65535).astype('uint16'))

    @staticmethod
    def correct_illumination(img, sigma = 30, double_correct = False, rescale_range = (1, 99), cutoff_threshold = None):
        cutoff1, cutoff2 = rescale_range
        img = skimage.util.img_as_float32(img)
        if cutoff_threshold is not None:
            if img.max() > (cutoff_threshold/65535):
                _img = img.copy()
                _img[_img > (cutoff_threshold/65535)] = 0
                p1 = np.percentile(_img, cutoff1)
                p99 = np.percentile(_img, cutoff2)
            else:
                p1 = np.percentile(img, cutoff1)
                p99 = np.percentile(img, cutoff2)
        else:
            p1 = np.percentile(img, cutoff1)
            p99 = np.percentile(img, cutoff2)

        img = skimage.exposure.rescale_intensity(img, 
                                                 in_range=(p1, p99), 
                                                 out_range=(0, 1))

        #calculate correction mask
        correction = gaussian(img, sigma)
        correction = invert(correction)
        correction = skimage.exposure.rescale_intensity(correction, 
                                                        out_range = (0,1))

        correction_lows =  np.where(img > 0.5, 0, img) * correction
        img_corrected = skimage.exposure.rescale_intensity(img + correction_lows,
                                                           out_range = (0,1))

        if double_correct:
            correction_mask_highs = invert(correction)
            correction_mask_highs_02 = skimage.exposure.rescale_intensity(np.where(img_corrected < 0.5, 0, img_corrected)*correction_mask_highs)
            img_corrected_double = skimage.exposure.rescale_intensity(img_corrected - 0.25*correction_mask_highs_02)
            
            return((img_corrected_double * 65535).astype('uint16'))
        else:
            return((img_corrected * 65535).astype('uint16'))
    
    def read(self, series, c):
        img = super().read(series, c)

        #check rescale_range type and set rescale_range accordingly
        if type(self.rescale_range) is dict:
            rescale_range = self.rescale_range[c]
        else:
            rescale_range = self.rescale_range
        if self.do_rescale == False or self.do_rescale == "full_image":
            return img
        elif self.do_rescale == "partial":
            if c not in self.no_rescale_channel:
                return self.rescale(img, rescale_range = rescale_range) 
            else:
                return img
        else:
            if c == self.WGAchannel:
                return self.correct_illumination(img, rescale_range = rescale_range)
            if c == "WGAbackground":
                return self.correct_illumination(img, double_correct = True, rescale_range = rescale_range)
            else:
                return self.rescale(img, rescale_range)  
            
from ashlar.reg import BioformatsReader

class BioformatsReaderRescale(BioformatsReader):

        def __init__(self, path, plate=None, well=None, do_rescale=False, no_rescale_channel = ["Alexa488"], rescale_range = (1, 99)):
            super().__init__(path, plate, well)
            self.do_rescale = do_rescale
            self.no_rescale_channel = no_rescale_channel
            self.rescale_range = rescale_range

        @staticmethod    
        def rescale(img, rescale_range = (1, 99)):
            img = skimage.util.img_as_float32(img)
            cutoff1, cutoff2 = rescale_range
            
            if img.max() > (40000/65535):
                _img = img.copy()
                _img[_img > (10000/65535)] = 0
                p1 = np.percentile(_img, cutoff1)
                p99 = np.percentile(_img, cutoff2)
            else:
                p1 = np.percentile(img, cutoff1)
                p99 = np.percentile(img, cutoff2)
            
            img = skimage.exposure.rescale_intensity(img, 
                                                    in_range=(p1, p99), 
                                                    out_range=(0, 1))
            return((img * 65535).astype('uint16'))

        def read(self, series, c):
            self.metadata._reader.setSeries(self.metadata.active_series[series])
            index = self.metadata._reader.getIndex(0, c, 0)
            byte_array = self.metadata._reader.openBytes(index)
            dtype = self.metadata.pixel_dtype
            shape = self.metadata.tile_size(series)
            img = np.frombuffer(byte_array.tostring(), dtype=dtype).reshape(shape)

            if not self.do_rescale:
                return img
            elif self.do_rescale == "partial":
                if c not in self.no_rescale_channel:
                    return self.rescale(img, self.rescale_range)
                else:
                    return img
            else:
                return self.rescale(img, self.rescale_range)