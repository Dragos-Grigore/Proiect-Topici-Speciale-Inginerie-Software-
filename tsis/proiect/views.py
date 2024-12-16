from django.shortcuts import render
from django.http import JsonResponse

from lens import Lens, LensProcessor
from PIL import Image
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, BloomTokenizerFast, BloomForCausalLM
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

    # If you want to run on CPU: just remove 'cuda:0'
    # For running on GPU: if you are on PC, type: "cuda" instead of "cuda:0". Use "cuda:0" if you are on laptop
    # For GPT - 2 you can just leave it like it is: device = 0

    if selected_llm == "Flan T5 Base":
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        LLM_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        LLM_model = LLM_model.to("cuda:0")

        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids.to("cuda:0")
        outputs = LLM_model.generate(input_ids)
        answer = tokenizer.decode(outputs[0])
    elif selected_llm == "Flan T5 Small":
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", truncation_side='left', padding=True)
        LLM_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        LLM_model = LLM_model.to("cuda:0")

        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids.to("cuda:0")
        outputs = LLM_model.generate(input_ids)
        answer = tokenizer.decode(outputs[0])
    elif selected_llm == "GPT-2":
        generator = pipeline('text-generation', model='gpt2', device = 0)

        answer = generator(samples["prompts"], max_length=250, num_return_sequences=1)[0][0]['generated_text']
        index = answer.find("Short Answer:") + len("Short Answer:")
        answer = answer[index:].strip()

    elif selected_llm == "BLOOM-Small":
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
        LLM_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
        LLM_model = LLM_model.to("cuda:0")

        input_ids = tokenizer(samples["prompts"], return_tensors="pt").input_ids.to("cuda:0")
        outputs = LLM_model.generate(input_ids)
        answer = tokenizer.decode(outputs[0], skip_special_tokens = True)
        index = answer.find("Short Answer:") + len("Short Answer:")
        answer = answer[index:].strip()
    return answer