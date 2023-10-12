from cae.model.reimagining_price_trends import create_model


def main():
    model, optimizer, loss_fn = create_model()
    for epoch in range(10):
        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = loss_fn(outputs, labels_onehot)

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero out any existing gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
