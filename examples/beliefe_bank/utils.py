import torch

class Generator(torch.nn.Module):

    def __init__(self,tokenizer,model):
        super(Generator, self).__init__()
        self.tokenizer=tokenizer
        self.model=model

    def forward(self,name,sentence):


        input_string = "$answer$ ; $mcoptions$  = (A) no (B) yes ; $question$ = "+name+" "+ sentence.replace(","," ").replace("IsA","is a").replace("CapableOf","is capable of")
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt")

        output = self.model.generate(input_ids, max_length=200,output_scores =True,return_dict_in_generate=True)
        return torch.Tensor((output.scores[6][0][150],output.scores[6][0][4273]))