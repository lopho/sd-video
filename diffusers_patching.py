import torch

def patch_diffusers_transformer_checkpointing(model: torch.nn.Module):
    from diffusers.models.attention import BasicTransformerBlock
    def transformer_checkpoint(m):
        if isinstance(m, BasicTransformerBlock):
            m._forward = m.forward
            def checkpointed(
                    hidden_states,
                    attention_mask = None,
                    encoder_hidden_states = None,
                    encoder_attention_mask = None,
                    timestep = None,
                    cross_attention_kwargs = None,
                    class_labels = None
            ):
                return torch.utils.checkpoint.checkpoint(m._forward,
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        cross_attention_kwargs,
                        class_labels
                )
            m.forward = checkpointed
    model.apply(transformer_checkpoint)

