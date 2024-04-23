for epoch in range(epochs):
    for i, (sketches, photos) in enumerate(dataloader):
        real_label = torch.ones(batch_size, 1)
        fake_label = torch.zeros(batch_size, 1)

        optimizer_dis.zero_grad()
        output = discriminator(photos)
        loss_real = criterion(output, real_label)

        noise = torch.randn(batch_size, noise_dim)
        fake_images = generator(noise)

        output = discriminator(fake_images.detach())
        loss_fake = criterion(output, fake_label)

        loss_dis = (loss_real + loss_fake) / 2
        loss_dis.backward()
        optimizer_dis.step()

        optimizer_gen.zero_grad()
        output = discriminator(fake_images)
        loss_gen = criterion(output, real_label)
        loss_gen.backward()
        optimizer_gen.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] Output [{output}]"
                  f"Loss D: {loss_dis.item():.4f}, Loss G: {loss_gen.item():.4f}")
