import torch
import torch.nn as nn
import torch.optim as optim


class RegX:
    def __init__(self, model, criterion, optimizer, alpha, target_layer, feature_importance_func):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.alpha = alpha
        self.target_layer = target_layer
        self.feature_importance_func = feature_importance_func

    def compute_tc_loss(self, feature_importance_scores):
        batch_size, latent_dim = feature_importance_scores.shape

        # Calculate log probability density of Gaussian distribution
        def log_density_gaussian(x, mu, var):
            return -0.5 * (torch.log(2 * torch.pi * var) + (x - mu) ** 2 / var).sum(dim=1)

        mu = feature_importance_scores.mean(dim=0)
        var = feature_importance_scores.var(dim=0)

        z = feature_importance_scores

        mat_log_q_z = log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                           mu.view(1, batch_size, latent_dim),
                                           var.view(1, batch_size, latent_dim))

        # Compute TC loss
        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)
        tc_loss = (log_q_z - log_prod_q_z).mean()

        return tc_loss

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            for inputs, targets in data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Obtain feature importance scores for the target layer
                features = self.model.get_features_at_layer(inputs, self.target_layer)
                feature_importance_scores = self.feature_importance_func(features, inputs)

                # Compute regularization term
                reg_loss = self.alpha * self.compute_tc_loss(feature_importance_scores)

                # Add regularization term to the loss
                loss += reg_loss

                # Backpropagation
                loss.backward()
                self.optimizer.step()

    def enhance_image(self, image):
        optimizer = optim.Adam([image.requires_grad_()], lr=0.01)
        criterion = torch.nn.MSELoss()

        for _ in range(100):  # Example: perform 100 optimization steps
            optimizer.zero_grad()
            output = self.model(image)
            feature_importance = self.feature_importance_function(output)
            loss = criterion(feature_importance, torch.zeros_like(feature_importance))
            loss.backward()
            optimizer.step()

        return image


class AdvX:
    def __init__(self, model, criterion, optimizer, beta, gamma, target_layer, feature_importance_func,
                 adversarial_generator):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.beta = beta
        self.gamma = gamma
        self.target_layer = target_layer
        self.feature_importance_func = feature_importance_func
        self.adversarial_generator = adversarial_generator

    def compute_adv_loss(self, inputs, adv_inputs, feature_importance_scores):
        # Compute the L1-norm of the adversarial perturbation for each feature
        adversarial_norms = torch.norm(adv_inputs - inputs, p=1, dim=1)

        # Compute the weighted sum of L1-norms using feature importance scores
        weighted_norms = torch.sum(feature_importance_scores * adversarial_norms.unsqueeze(1), dim=0)

        # Compute the L1-norm of the difference between original and perturbed feature representations
        feature_representation_diff = self.model.get_features_at_layer(inputs, self.target_layer) - \
                                      self.model.get_features_at_layer(adv_inputs, self.target_layer)
        feature_representation_l1_norm = torch.norm(feature_representation_diff, p=1)

        # Compute the adversarial loss
        adv_loss = weighted_norms - self.gamma * feature_representation_l1_norm

        return adv_loss

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            for inputs, targets in data_loader:
                self.optimizer.zero_grad()

                # Forward pass with original inputs
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Obtain feature importance scores for the target layer
                features = self.model.get_features_at_layer(inputs, self.target_layer)
                feature_importance_scores = self.feature_importance_func(features, inputs)

                # Generate adversarial inputs
                adv_inputs = self.adversarial_generator(inputs)

                # Compute adversarial loss
                adv_loss = self.compute_adv_loss(inputs, adv_inputs, feature_importance_scores)

                # Add adversarial loss to the original loss
                loss -= self.beta * adv_loss

                # Backpropagation
                loss.backward()
                self.optimizer.step()

    def enhance_image(self, image):
        optimizer = optim.Adam([image.requires_grad_()], lr=0.01)
        criterion = torch.nn.MSELoss()

        for _ in range(100):  # Example: perform 100 optimization steps
            optimizer.zero_grad()
            output = self.model(image)
            feature_importance = self.feature_importance_function(output)

            # Generate adversarial perturbation
            perturbation = torch.randn_like(image)
            perturbed_image = image + perturbation

            perturbed_output = self.model(perturbed_image)
            perturbed_feature_importance = self.feature_importance_function(perturbed_output)

            adv_loss = torch.sum(
                feature_importance * torch.norm(perturbation, p=1, dim=(1, 2, 3))) - self.gamma * torch.norm(
                output - perturbed_output, p=1)
            loss = criterion(feature_importance, torch.zeros_like(feature_importance)) + self.beta * adv_loss
            loss.backward()
            optimizer.step()

        return image


class Enhancer(RegX, AdvX):
    def __init__(self, encoder, decoder, feature_importance_function, regx=False, advx=False):
        self.encoder = encoder
        self.decoder = decoder
        self.feature_importance_function = feature_importance_function
        self.regx = regx
        self.advx = advx

    def enhance_data(self, data):
        optimizer = optim.Adam([data.requires_grad_()], lr=0.01)

        for _ in range(100):  # Example: perform 100 optimization steps
            optimizer.zero_grad()
            latent_representation = self.encoder(data)
            feature_importance = self.feature_importance_function(latent_representation)
            output = self.decoder(latent_representation)

            if self.regx:
                loss = self.compute_regx_loss(feature_importance)
            elif self.advx:
                loss = self.compute_advx_loss(data, output, feature_importance)
            else:
                # Default to MSELoss if neither RegX nor AdvX is specified
                criterion = torch.nn.MSELoss()
                loss = criterion(feature_importance, torch.zeros_like(feature_importance))

            loss.backward()
            optimizer.step()

        return data

