import os
import torch
import torch.nn as nn
import torch.optim as optim

# define the generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
    
    def forword(self, x):
        return self.generator(x)

# define the discriminator    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forword(self, x):
        return self.discriminator(x)

# define the loss function
def loss_fn(logits, labels):
    # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    return nn.BCELoss()(logits, labels)

# define the optimizer
def get_optimizer(model, learning_rate):
    return optim.Adam(model.parameters(), lr=learning_rate)

# define the data loader
def get_data_loader(data, batch_size):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

# define the train function
def train():
    # Model parameters
    g_input_size = 1      # Random noise dimension coming into generator, per output vector
    g_hidden_size = 5     # Generator complexity
    g_output_size = 1     # Size of generated output vector
    d_input_size = 500    # Minibatch size - cardinality of distributions
    d_hidden_size = 10    # Discriminator complexity
    d_output_size = 1     # Single dimension for 'real' vs. 'fake' classification
    minibatch_size = d_input_size

    g_learning_rate = 1e-3
    d_learning_rate = 1e-3

    num_epochs = 5000
    print_interval = 100
    g_steps = 20
    d_steps = 20

    # Training parameters
    G = Generator(g_input_size, g_hidden_size, g_output_size)
    D = Discriminator(d_input_size, d_hidden_size, d_output_size)

    # Loss function and optimizer
    criterion = loss_fn()
    g_optimizer = get_optimizer(G, g_learning_rate)
    d_optimizer = get_optimizer(D, d_learning_rate)

    # Data loader
    data_loader = get_data_loader(mnist, minibatch_size)

    # Start training
    for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            # 1. Train the discriminator on real and fake data
            for _ in range(d_steps):
                # 1A: Train D on real
                d_optimizer.zero_grad()

                # 1A1: Compute the probability of real images being real
                d_real_data = D(real_batch)
                d_real_decision = d_real_data
                d_real_error = criterion(d_real_decision, torch.ones_like(d_real_decision))  # ones = true

                # 1A2: Compute the probability of fake images being real
                d_gen_input = torch.randn(minibatch_size, g_input_size)
                d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_decision = D(d_fake_data)
                d_fake_error = criterion(d_fake_decision, torch.zeros_like(d_fake_decision))  # zeros = fake
                
                # 1A3: Backpropagate errors
                d_error = d_real_error + d_fake_error
                d_error.backward()
                d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

                # 1B: Train D on fake
                d_optimizer.zero_grad()

                # 1B1: Generate fake data
                d_gen_input = torch.randn(minibatch_size, g_input_size)
                d_fake_data = G(d_gen_input)

                # 1B2: Compute the probability of fake images being real
                d_fake_decision = D(d_fake_data)
                d_fake_error = criterion(d_fake_decision, torch.ones_like(d_fake_decision))  # ones = true

                # 1B3: Backpropagate errors
                

if __name__ == '__main__':
    if not os.path.exists('./img'):
        os.mkdir('./img')
    
    from torchvision import datasets, transforms
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1,), (0.5,))
    ])
    mnist = datasets.MNIST(
        root='./mnist/', train=True, transform=img_transform, download=False
    )
    train()