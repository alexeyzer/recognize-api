import grpc
from concurrent import futures
import gen.python.api.recognize.v1.recognize_api_pb2 as pb
import gen.python.api.recognize.v1.recognize_api_pb2_grpc as pb_grpc
import torch
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import io

MODEL_PATH = "./stage-1_resnet34_1.pkl"
CLASSES_PATH = "./classes.txt"


class ClassificationModel():
    def __init__(self):
        return

    def load(self, model_path, labels_path, eval=False):
        self.model = torch.load(model_path)
        self.model = nn.Sequential(self.model)

        self.labels = open(labels_path, 'r').read().splitlines()

        if eval:
            print(self.model.eval())
        return

    def predict(self, image):
        #img = Image.open(image_path)
        img = Image.open(io.BytesIO(image))

        test_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])
                                              ])

        image_tensor = test_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        inp = Variable(image_tensor)
        output = self.model(inp)
        index = output.data.cpu().numpy().argmax()
        return self.labels[index]


class RecognizeApiService(pb_grpc.RecognizeApiService):

    def __init__(self, *args, **kwargs):
        self.ClassificationClass = ClassificationModel()
        self.ClassificationClass.load(MODEL_PATH, CLASSES_PATH)
        print("loaded")
        pass

    def recognizePhoto(self, request, context):

        # get the string from the incoming request
        classs = self.ClassificationClass.predict(request.image)
        result = {'category': classs}

        return pb.recognizePhotoResponse(**result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_RecognizeApiServiceServicer_to_server(RecognizeApiService(), server)
    server.add_insecure_port('[::]:8082')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    print("app starting")
    serve()