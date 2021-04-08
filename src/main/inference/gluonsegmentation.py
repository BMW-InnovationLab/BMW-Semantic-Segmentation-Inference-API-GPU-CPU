import os
import mxnet as mx
import asyncio
import json
from PIL import Image, ImageDraw, ImageFont
from  inference.base_inference_engine import AbstractInferenceEngine
from inference.exceptions import InvalidModelConfiguration, InvalidInputData, ApplicationError
import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from mxnet import image
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms
import numpy    
import gluoncv
from gluoncv.utils.viz import get_color_pallete
import sys
from gluoncv.loss import *
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.data.pascal_voc.segmentation import VOCSegmentation 
from contextlib import contextmanager
from mxnet import ndarray as nd
import cv2
from io import BytesIO
import time


class InferenceEngine(AbstractInferenceEngine):

    def __init__(self, model_path):
        self.nclasses=0
        self.model="none"
        self.backbone="none"
        self.palette=[]
        self.font = ImageFont.truetype("/main/fonts/DejaVuSans.ttf", 20)
        super().__init__(model_path)
	
    def free(self):
    	
        pass
    """
    Function that loads models for inference 
    """
    def load(self):

        with open(os.path.join(self.model_path, 'configuration.json')) as f:
            data = json.load(f)
        try:

            self.set_model_configuration(data)
            self.validate_json_configuration(data)
        except ApplicationError as e:
            raise e
        self.nclasses=data["classes"]
        self.model=data["network"]
        self.backbone=data["backbone"]

        gpu_count=mx.util.get_gpu_count()
        if gpu_count>0:
            self.ctx1 = [mx.gpu()]
        else:
            self.ctx1= [mx.cpu()]
        VOCSegmentation.NUM_CLASS=self.nclasses
        self.net=get_segmentation_model(model=self.model,backbone=self.backbone,norm_layer=mx.gluon.nn.BatchNorm,norm_kwargs={},crop_size=480,aux=False,pretrained_base=False,ctx=self.ctx1)
        #self.net=get_model("fcn_resnet101_voc",pretrained=True,aux=False,crop_size=480,pretrained_base=True)
        self.net.initialize(force_reinit=True)
        net = DataParallelModel(self.net, self.ctx1, {})

        self.net.load_parameters(os.path.join(self.model_path,"model_best.params"),ctx=self.ctx1,ignore_extra=True)
        infile = open(os.path.join(self.model_path,'palette.txt'),'r')
        for line in infile:
            self.palette.append(int(line.strip('\n')))
        self.labels=data["classesname"]
        self.background=None
        if "background" in self.labels:
            self.background=list(self.labels).index("background")

  

    """
    Function that decides if segments or a json response should be returned 
    """

    async def infer(self, input_data, draw, predict_batch):
        
        response=None
        if draw:
            try:
                
                await self.inference(input_data.file.read())
            except ApplicationError as e:
                raise e
        else:
           response= await self.processing(input_data.file.read())
           
           return response

    """
    Function that returns segments
    """
    async def inference2(self,input_data):
        start_time=time.time()
        model=self.net
        
        bytes_as_np_array = np.frombuffer(input_data.file.read(), dtype=np.uint8)
        img=image.imdecode(bytes_as_np_array,cv2.IMREAD_COLOR)
        img = test_transform(img,self.ctx1[0])
        mx.nd.waitall()

        output = model.predict(img)
        if self.model=='fcn':
           predict = mx.nd.squeeze(mx.nd.argmax(nd.softmax(output[0],axis=1), 1)).asnumpy()
        else: 
            argmax=mx.nd.argmax(output, 1)
            predict = mx.nd.squeeze(argmax)
            predict= predict.asnumpy()
        mx.nd.waitall()
        mask=predict
        mask=Image.fromarray(mask.astype('uint8'))
        mask.putpalette(self.palette)
        mask.save("/main/result.jpg","png")
        for ctx2 in self.ctx1:
            ctx2.empty_cache()
        print(time.time()-start_time)
    """ 
    Function that is called when infering multiple pictures 
    """       
    async def run_batch(self, input_data, draw, predict_batch):
        result_list = []
        for image in input_data:
            post_process = await self.infer(image, draw, predict_batch)
            if post_process is not None:
                result_list.append(post_process)
        return result_list
    
    """
    Function that returns segments pasted on top the original image 
    """

    async def inference(self,input_data):
        model=self.net
        bytes_as_np_array = np.frombuffer(input_data, dtype=np.uint8)
        img=image.imdecode(bytes_as_np_array,cv2.IMREAD_COLOR)
        img = test_transform(img,self.ctx1[0])
        output = model.predict(img)
        if self.model=='fcn':
           predict = mx.nd.squeeze(mx.nd.argmax(nd.softmax(output[0],axis=1), 1)).asnumpy()
        else:
           predict = mx.nd.squeeze(mx.nd.argmax(nd.softmax(output,axis=1), 1)).asnumpy()
        
        mask=predict

        mask=Image.fromarray(mask.astype('uint8'))
        mask.putpalette(self.palette)        
        img=Image.open(BytesIO(input_data))
        img=img.convert(mode='RGBA')
        
        mask=mask.convert(mode="RGBA")
        mask=np.array(mask)

        mask[:,:,3]=200
        if self.background != None :
            red, green, blue=mask[:,:,0], mask[:,:,1], mask[:,:,2]
            transfer=(red == self.palette[self.background*3]) & (green == self.palette[self.background*3+1]) & (blue ==self.palette[self.background*3+2])
            mask[:,:,:4][transfer]=[0,0,0,0]
        mask=Image.fromarray(mask.astype('uint8'))
        
    
        img.paste(mask,mask)
        img.save('/main/result.jpg','PNG')
        for ctx2 in self.ctx1:
            ctx2.empty_cache()
        
    """
    Function that returns a json response after infernece 
    """

    async def processing(self, input_data):
        await asyncio.sleep(0.00001)
        model=self.net
        bytes_as_np_array = np.frombuffer(input_data, dtype=np.uint8)
        img=image.imdecode(bytes_as_np_array,cv2.IMREAD_COLOR)
        img = test_transform(img,self.ctx1[0])
        output = model.predict(img)
        if self.model=='fcn':
           predict = mx.nd.squeeze(mx.nd.argmax(nd.softmax(output[0],axis=1), 1)).asnumpy()
           confidences=nd.softmax(output[0],axis=1).asnumpy()
        else:
           predict = mx.nd.squeeze(mx.nd.argmax(nd.softmax(output,axis=1), 1)).asnumpy()
           confidences=nd.softmax(output,axis=1).asnumpy()

        mask=np.copy(predict)
        d={}
        d=numpy.unique(mask)

        
                    
        boxes = []
        mask=Image.fromarray(mask.astype('uint8'))
        mask.putpalette(self.palette)
        mask=mask.convert(mode="RGB")
        image1=cv2.cvtColor(np.array(mask),cv2.COLOR_RGB2BGR)
        response1=[]
        for key in d :
            if key != self.background:
                

            
                lower=np.array([self.palette[(int(key)*3)+2],self.palette[(int(key)*3)+1],self.palette[(int(key)*3)]])
                upper=np.array([self.palette[(int(key)*3)+2]+1,self.palette[(int(key)*3)+1]+1,self.palette[(int(key)*3)]+1])

                mask1=cv2.inRange(image1,lower,upper)
                # Finding Contours 
                # Use a copy of the image e.g. edged.copy() 
                # since findContours alters the image 
                contours, hierarchy = cv2.findContours(mask1,  
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                segments=[]
                for c in contours:
                    (x, y, w, h) = cv2.boundingRect(c)
                    if(len(np.array(c).flatten()) > 3):
                        boxes.append([x,y, x+w,y+h])
                        segments.append(np.array(c).flatten())
                confidence=[]
                confidence1=[]
                counting=0
                for b in boxes:
                    
                    ind=int(key)
                    response=dict([("class_name",str(self.labels[ind])),("ObjectClassId",ind),("bbox",b),("segment",list(segments[counting])),("confidence","0.998")])
                    response=json.loads(str(response).replace("\'", "\""))
                    response1.append(response)                  
                    response1=json.loads(str(response1).replace("\'", "\""))
                    confidence1=[]
                    counting=counting+1
                boxes=[]

        for ctx2 in self.ctx1:
            ctx2.empty_cache()


        
        return response1

                    




    def free(self):
        pass

    def validate_configuration(self):
        # check if weights and palette file exists
        if not os.path.exists(os.path.join(self.model_path, 'model_best.params')):
            raise InvalidModelConfiguration('model_best.params')
        if not os.path.exists(os.path.join(self.model_path, 'palette.txt')):
            raise InvalidModelConfiguration('palette.txt')
        return True
    
    #Load model configuration
    def set_model_configuration(self, data):
        self.configuration['type']=data['type']
        try:
            self.configuration['nclasses'] = data['classes']
            if not isinstance(self.configuration['nclasses'], int):
                raise InvalidModelConfiguration('nclasses field should be of type int')
        except ApplicationError as e:
            raise e
        except Exception as e:
            raise InvalidModelConfiguration('missing nclasses field in config.json')
        try:
            self.configuration['backbone'] = data['backbone']
            if not isinstance(self.configuration['backbone'], str):
                raise InvalidModelConfiguration('backbone field should be of type string')
        except ApplicationError as e:
            raise e
        except Exception as e:
            raise InvalidModelConfiguration('missing backbone field in config.json')
        try:
            self.configuration['model'] = data['network']
            if not isinstance(self.configuration['model'], str):
                raise InvalidModelConfiguration('model field should be of type string')
        except ApplicationError as e:
            raise e
        except Exception as e:
            raise InvalidModelConfiguration('missing model field in config.json')
        try:
            self.configuration['classesname'] = data['classesname']
            if not isinstance(self.configuration['classesname'], list):
                raise InvalidModelConfiguration('classes field should be of type list')
        except ApplicationError as e:
            raise e
        except Exception as e:
            raise InvalidModelConfiguration('missing classesname field in config.json')

    #Validate the json configuration
    def validate_json_configuration(self, data):
        try:
            if not isinstance(data['classes'], int):
                raise InvalidModelConfiguration('classes field should be of type int')
        except ApplicationError as e:
            raise e
        except Exception as e:
            raise InvalidModelConfiguration('missing confidence field in config.json')
        try:
            if not isinstance(data['classesname'], list):
                raise InvalidModelConfiguration('classesname field should be of type int')
        except ApplicationError as e:
            raise e
        except Exception as e:
            raise InvalidModelConfiguration('missing classesname field in config.json')
