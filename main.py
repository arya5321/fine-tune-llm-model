import os
os.environ['GRADIENT_WORKSPACE_ID']='b54cf698-a547-4daa-a873-b83a54cdd134_workspace'
os.environ['GRADIENT_ACCESS_TOKEN']='hnV3DOellXEMihUI6GsU2UdkgJlRKGw4'

from re import S
from gradientai import Gradient

def main():
  gradient = Gradient()
  base_model =gradient.get_base_model(base_model_slug='nous-hermes2')
  new_model=base_model.create_model_adapter(name= 'aryamodel')

  print('created model with id:',new_model.id)
  sample_query = "### instruction: who is Arya?\n\n ###Response:"
  print("Asking:",sample_query)

##before finetuning

  completion = new_model.complete(query=sample_query,max_generated_token_count=100).generated_output
  print('Generated(before fine tuning):', completion)


  samples=[
      {"inputs":"### instruction: who is arya? \n\n### Response: Arya is a student who is currently pursuing btech in Artificial intelligence and Datascience."},
      {"inputs":"### instruction: who is arya? \n\n### Response: Arya is an introvert.however she is extrovert arround her friends"}
  ]


  num_epochs=3
  count=0
  while count<num_epochs:
    print("fine tuning the model with iteration:",count+1)
    new_model.fine_tune(samples=samples)
    count=count+1

  #after fine tuning
  completion = new_model.complete(query=sample_query,max_generated_token_count=100).generated_output
  print('Generated(after fine tuning):', completion)
    
if __name__=="__main__":
  main()
