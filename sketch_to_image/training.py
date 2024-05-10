# Training loop
for epoch in range(num_epochs):
    for batch_idx, (sketch, real_image) in enumerate(data_loader):
        optimizer_D.zero_grad()

        real_combined = torch.cat([sketch, real_image], dim=1)
        real_output = discriminator(real_combined)
        real_label = torch.ones_like(real_output)
        loss_real = adversarial_loss(real_output, real_label)

        fake_image = generator(sketch)
        fake_combined = torch.cat([sketch, fake_image], dim=1)
        fake_output = discriminator(fake_combined)
        fake_label = torch.zeros_like(fake_output)
        loss_fake = adversarial_loss(fake_output, fake_label)

        # Discriminator loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Training the generator
        optimizer_G.zero_grad()

        fake_image = generator(sketch)
        fake_combined = torch.cat([sketch, fake_image], dim=1)
        fake_output = discriminator(fake_combined)

        generator_label = torch.ones_like(fake_output)
        loss_adv = adversarial_loss(fake_output, generator_label)

        loss_rec = reconstruction_loss(fake_image, real_image)

        # Generator loss
        lambda_recon = 100
        loss_G = loss_adv + (lambda_recon * loss_rec)
        loss_G.backward()
        optimizer_G.step()
