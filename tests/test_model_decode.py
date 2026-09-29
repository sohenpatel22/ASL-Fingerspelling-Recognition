import torch

from asl.config import ModelConfig
from asl.decode import beam_search, greedy_decode
from asl.model import ASLConformerSeq2Seq


def test_released_architecture_parameter_count(vocab):
    model = ASLConformerSeq2Seq(ModelConfig(), vocab.vocab_size, vocab.pad_idx)
    assert 27.5e6 < model.num_parameters() < 28.1e6


def test_state_dict_keys_match_released_checkpoints(vocab):
    keys = set(ASLConformerSeq2Seq(ModelConfig(), vocab.vocab_size, vocab.pad_idx).state_dict())
    for expected in (
        "encoder.proj.0.weight", "encoder.posenc.pe", "encoder.layers.5.ff1.1.weight",
        "encoder.layers.0.conv.2.weight", "encoder.layers.0.attn.in_proj_weight",
        "decoder.embed.weight", "decoder.proj_emb.weight", "decoder.fc_out.weight",
        "decoder.decoder.layers.5.multihead_attn.in_proj_weight",
    ):
        assert expected in keys, expected


def test_forward_shapes(tiny_model, tiny_cfg, vocab):
    x = torch.randn(3, 84, tiny_cfg.model.max_seq_len)
    y = torch.randint(0, vocab.n_classes, (3, 10))
    assert tiny_model(x, y).shape == (3, 10, vocab.vocab_size)
    assert tiny_model.encoder(x).shape == (3, tiny_cfg.model.max_seq_len, tiny_cfg.model.d_model)


def test_causal_decoder_ignores_future_tokens(tiny_model, tiny_cfg, vocab):
    x = torch.randn(1, 84, tiny_cfg.model.max_seq_len)
    a = torch.tensor([[vocab.start_idx, 3, 4, 5]])
    b = torch.tensor([[vocab.start_idx, 3, 9, 9]])
    with torch.no_grad():
        la, lb = tiny_model(x, a), tiny_model(x, b)
    torch.testing.assert_close(la[:, :2], lb[:, :2])  # positions before the change agree


def test_greedy_decode_shapes_and_termination(tiny_model, tiny_cfg, vocab):
    x = torch.randn(4, 84, tiny_cfg.model.max_seq_len)
    out = greedy_decode(tiny_model, x, vocab, max_len=12)
    assert len(out) == 4 and all(len(o) <= 11 for o in out)


def test_beam_width_one_equals_greedy(tiny_model, tiny_cfg, vocab):
    torch.manual_seed(0)
    x = torch.randn(1, 84, tiny_cfg.model.max_seq_len)
    greedy = greedy_decode(tiny_model, x, vocab, max_len=10)[0]
    beam, _ = beam_search(tiny_model, x[0], vocab, beam_width=1, max_len=10, length_penalty=0.0)
    assert vocab.decode(beam) == vocab.decode(greedy)


def test_beam_search_output_contract(tiny_model, tiny_cfg, vocab):
    x = torch.randn(84, tiny_cfg.model.max_seq_len)
    tokens, score = beam_search(tiny_model, x, vocab, beam_width=3, max_len=8)
    assert tokens[0] == vocab.start_idx and len(tokens) <= 8 and score <= 0.0
