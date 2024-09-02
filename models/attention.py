cd 
        self.encoder = nn.ModuleList()

        modules = []

        for i in range(layers):
            mha = TransformerBlock(feat_size, nhum_head, dim, hiden_size)
            modules.append(mha)

        self.encoder = nn.Sequential(*modules)

        self.out_size = out_size

        if self.out_size != 0:
            self.out_fc = nn.Linear(feat_size, out_size)

    def forward(self, x_seq):
        
        x_ffn = self.encoder(x_seq)
        if self.out_size != 0:
            x_ffn = self.out_fc(x_ffn)
    
        return x_ffn


if __name__ == '__main__':
    net = MHASA(2048, 4, 256)

    x = torch.rand(5, 2048, 8, 8)

    y, atten = net(x)

    print(y.shape, atten.shape)