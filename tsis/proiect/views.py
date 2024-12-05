from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

from lens import Lens, LensProcessor
from PIL import Image
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def multimodal_page(request):
    return render(request, 'proiect.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        
        filename = fs.save(image.name, image)

        message = analyze_image(image)
        uploaded_image_url = fs.url(filename)
        absolute_image_path = fs.path(filename)

        return JsonResponse({
            'message': message,
            'image_url': uploaded_image_url,
            'image_path': absolute_image_path,
        })
    
    return JsonResponse({'message': 'Invalid request.'}, status=400)

def analyze_image(image):
    image = Image.open(image).convert('RGB')
    question = "What is the image about?"

    lens = Lens()
    processor = LensProcessor()
    with torch.no_grad():
        samples = processor([image],[question])
        lens(samples)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small",truncation_side = 'left',padding = True)
    LLM_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
    outputs = LLM_model.generate(input_ids)

    return tokenizer.decode(outputs[0])