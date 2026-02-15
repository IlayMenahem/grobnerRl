"""Test the new Grain-based replay buffer with file storage."""

import numpy as np
import tempfile
import os

from grobnerRl.training.shared import (
    Experience,
    GrainReplayBuffer,
)


def test_grain_replay_buffer_basic():
    """Test basic replay buffer operations with file storage."""
    # Create a temporary directory for storage
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create replay buffer with file storage and workers
        buffer = GrainReplayBuffer(
            max_size=100,
            storage_dir=temp_dir,
            worker_count=2,
            worker_buffer_size=4,
        )
        
        # Create some dummy experiences
        experiences = []
        for i in range(10):
            ideal = tuple(
                np.random.randn(np.random.randint(3, 8), 4).astype(np.float32)
                for _ in range(5)
            )
            selectables = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
            policy = np.random.rand(25).astype(np.float32)
            policy = policy / policy.sum()
            value = float(np.random.rand())
            
            experiences.append(
                Experience(
                    ideal=ideal,
                    selectables=selectables,
                    policy=policy,
                    value=value,
                    num_polys=5,
                )
            )
        
        # Add experiences
        buffer.add(experiences)
        assert len(buffer) == 10
        
        # Test file was created
        assert os.path.exists(buffer.storage_path)
        
        # Sample a batch
        obs, policies, values, masks = buffer.sample_batched(batch_size=4)
        assert obs["ideals"].shape[0] == 4
        assert policies.shape[0] == 4
        assert values.shape[0] == 4
        assert masks.shape[0] == 4
        
        print("✓ Basic replay buffer test passed!")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


def test_grain_replay_buffer_fifo():
    """Test FIFO eviction when buffer is full."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        buffer = GrainReplayBuffer(
            max_size=5,
            storage_dir=temp_dir,
            worker_count=1,
            worker_buffer_size=2,
        )
        
        # Add more experiences than max_size
        for batch_idx in range(3):
            experiences = []
            for i in range(3):
                ideal = tuple(
                    np.random.randn(3, 4).astype(np.float32) * (batch_idx + 1)
                    for _ in range(5)
                )
                selectables = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
                policy = np.random.rand(25).astype(np.float32)
                policy = policy / policy.sum()
                value = float(batch_idx + i)
                
                experiences.append(
                    Experience(
                        ideal=ideal,
                        selectables=selectables,
                        policy=policy,
                        value=value,
                        num_polys=5,
                    )
                )
            
            buffer.add(experiences)
        
        # Should have max_size experiences
        assert len(buffer) == 5
        print("✓ FIFO eviction test passed!")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_grain_replay_buffer_iteration():
    """Test iterating through the buffer with DataLoader."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        buffer = GrainReplayBuffer(
            max_size=20,
            storage_dir=temp_dir,
            worker_count=2,
            worker_buffer_size=4,
        )
        
        # Add experiences
        experiences = []
        for i in range(15):
            ideal = tuple(
                np.random.randn(3, 4).astype(np.float32)
                for _ in range(5)
            )
            selectables = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
            policy = np.random.rand(25).astype(np.float32)
            policy = policy / policy.sum()
            value = float(i)
            
            experiences.append(
                Experience(
                    ideal=ideal,
                    selectables=selectables,
                    policy=policy,
                    value=value,
                    num_polys=5,
                )
            )
        
        buffer.add(experiences)
        
        # Iterate through batches
        batch_count = 0
        total_samples = 0
        for obs, policies, values, masks in buffer.iter_dataset(batch_size=4, shuffle=True):
            batch_count += 1
            batch_size = obs["ideals"].shape[0]
            total_samples += batch_size
            assert batch_size <= 4
            assert policies.shape[0] == batch_size
        
        # Should have seen all samples (15 samples with batch_size=4 -> 4 batches)
        assert total_samples == 15
        assert batch_count == 4  # 3 full batches + 1 partial
        
        print("✓ Iteration test passed!")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_grain_replay_buffer_basic()
    test_grain_replay_buffer_fifo()
    test_grain_replay_buffer_iteration()
    print("\n✅ All tests passed!")
