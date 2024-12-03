from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

def mutlimodal_page(request):
    
    return render(request, 'proiect.html')

def upload_data(request):
    
    if request.method == 'POST':
        selected_prompt = request.POST.get("prompt-select")  
        selected_llm = request.POST.get("llm-select")  
        image = request.FILES.get('image')
        
        if image:
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_image_url = fs.url(filename)
            absolute_image_path = fs.path(filename)
            
            
            return JsonResponse({
                'message': 'Image uploaded successfully!',
                'image_url': uploaded_image_url,
                'image_path': absolute_image_path,
                'selected_llm': selected_llm,
                'selected_prompt': selected_prompt,
            })

    
    return JsonResponse({'message': 'Invalid request.'}, status=400)




