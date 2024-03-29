import torch

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.call_time = 0

    def attack(self, epsilon=1., emb_name='.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if self.call_time == 0:
                    print(f'Attack {name}')
                    
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if self.call_time == 0:
                    print(f'Restore {name}')
                    self.call_time += 1
                
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}