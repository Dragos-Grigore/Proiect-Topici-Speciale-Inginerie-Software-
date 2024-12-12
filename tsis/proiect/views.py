from django.shortcuts import render
from django.http import JsonResponse

from lens import Lens, LensProcessor
from PIL import Image
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re
def multimodal_page(request):
    return render(request, 'proiect.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image') and request.POST.get('selected_llm') and request.POST.get('llm_query'):
        image = request.FILES['image']
        selected_llm = request.POST.get('selected_llm')
        llm_query = request.POST.get('llm_query')

        answer = analyze_image(image, selected_llm, llm_query)
        cleaned_text = re.sub(r"<.*?>", "", answer).strip()
    # Capitalize the first letter and ensure proper punctuation
        result = cleaned_text[0].upper() + cleaned_text[1:]
        if not result.endswith('.'):
            result += '.'
        answer=result
        return JsonResponse({
            'answer': answer
        })
    
    return JsonResponse({'answer': 'Invalid request.'}, status=400)

def analyze_image(image, selected_llm, llm_query):
    image = Image.open(image).convert('RGB')

    lens = Lens()
    processor = LensProcessor()
    with torch.no_grad():
        samples = processor([image], [llm_query])
        lens(samples)

    if selected_llm == "Flan T5 Base":
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        LLM_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
        outputs = LLM_model.generate(input_ids)
        answer = tokenizer.decode(outputs[0])
    elif selected_llm == "Flan T5 Small":
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side='left', padding=True)
        LLM_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
        outputs = LLM_model.generate(input_ids)
        answer = tokenizer.decode(outputs[0])
    elif selected_llm == "GPT-2":
        generator = pipeline('text-generation', model='gpt2')

        answer = generator(samples["prompts"], max_length=250, num_return_sequences=1)[0][0]['generated_text']
        index = answer.find("Short Answer:") + len("Short Answer:")
        answer = answer[index:].strip()
    elif selected_llm == "BART Base":
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", truncation_side='left', padding=True)
        LLM_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids
        outputs = LLM_model.generate(input_ids)
        answer = tokenizer.decode(outputs[0])

    return answer