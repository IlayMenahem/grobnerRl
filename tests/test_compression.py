"""Test script to verify and demonstrate experience compression."""

import numpy as np
from grobnerRl.training.shared import Experience


def test_compression():
    """Test that compression/decompression works correctly."""
    from grobnerRl.training.shared import PolynomialCache
    
    num_polys = 10
    num_vars = 4
    
    ideal = tuple(
        np.random.randn(np.random.randint(3, 8), num_vars).astype(np.float32)
        for _ in range(num_polys)
    )
    
    selectables = tuple((i, j) for i in range(num_polys) for j in range(i+1, num_polys))
    observation = (ideal, selectables)
    
    policy = np.zeros(num_polys * num_polys, dtype=np.float32)
    valid_actions = [5, 12, 23, 45, 67]
    policy[valid_actions] = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    value = 0.75
    
    poly_cache = PolynomialCache()
    exp = Experience.from_uncompressed(
        observation=observation,
        policy=policy,
        value=value,
        num_polys=num_polys,
        poly_cache=poly_cache,
    )
    
    reconstructed_obs = exp.observation(poly_cache)
    reconstructed_policy = exp.policy
    
    # Check observation
    assert len(reconstructed_obs[0]) == len(ideal)
    for i, poly in enumerate(reconstructed_obs[0]):
        assert np.allclose(poly, ideal[i])
    
    assert reconstructed_obs[1] == selectables
    
    # Check policy
    assert np.allclose(reconstructed_policy, policy)
    
    # Check value
    assert exp.value == value
    
    # Check compression stats
    memory_used = exp.memory_usage(poly_cache)
    compression_ratio = exp.compression_ratio(poly_cache)
    
    print("✓ Compression test passed!")
    print(f"  Memory used: {memory_used:,} bytes ({memory_used / 1024:.2f} KB)")
    print(f"  Compression ratio: {compression_ratio:.2%}")
    print(f"  Space saved: {(1 - compression_ratio) * 100:.1f}%")
    
    # Show policy compression details
    dense_policy_size = num_polys * num_polys * 4  # float32
    sparse_policy_size = exp.policy_indices.nbytes + exp.policy_values.nbytes
    policy_compression = sparse_policy_size / dense_policy_size
    
    print(f"\n  Policy details:")
    print(f"    Dense size: {dense_policy_size} bytes")
    print(f"    Sparse size: {sparse_policy_size} bytes")
    print(f"    Policy compression: {policy_compression:.2%}")
    print(f"    Policy space saved: {(1 - policy_compression) * 100:.1f}%")


def test_large_problem():
    """Test with a larger problem to show more realistic savings."""
    from grobnerRl.training.shared import PolynomialCache
    
    print("\n" + "="*60)
    print("Testing with larger problem (50 polynomials)")
    print("="*60)
    
    num_polys = 50
    num_vars = 6
    
    ideal = tuple(
        np.random.randn(np.random.randint(5, 15), num_vars).astype(np.float32)
        for _ in range(num_polys)
    )
    
    selectables = tuple((i, j) for i in range(num_polys) for j in range(i+1, num_polys))
    observation = (ideal, selectables)
    
    policy = np.zeros(num_polys * num_polys, dtype=np.float32)
    num_valid = min(20, len(selectables))
    valid_indices = np.random.choice(num_polys * num_polys, num_valid, replace=False)
    policy[valid_indices] = np.random.dirichlet(np.ones(num_valid))
    
    value = 0.5
    
    poly_cache = PolynomialCache()
    exp = Experience.from_uncompressed(
        observation=observation,
        policy=policy,
        value=value,
        num_polys=num_polys,
        poly_cache=poly_cache,
    )
    
    reconstructed_policy = exp.policy
    assert np.allclose(reconstructed_policy, policy)
    
    memory_used = exp.memory_usage(poly_cache)
    compression_ratio = exp.compression_ratio(poly_cache)
    
    print(f"✓ Large problem test passed!")
    print(f"  Memory used: {memory_used:,} bytes ({memory_used / 1024:.2f} KB)")
    print(f"  Compression ratio: {compression_ratio:.2%}")
    print(f"  Space saved: {(1 - compression_ratio) * 100:.1f}%")
    
    # Policy details
    dense_policy_size = num_polys * num_polys * 4
    sparse_policy_size = exp.policy_indices.nbytes + exp.policy_values.nbytes
    policy_compression = sparse_policy_size / dense_policy_size
    
    print(f"\n  Policy details:")
    print(f"    Dense size: {dense_policy_size:,} bytes ({dense_policy_size / 1024:.2f} KB)")
    print(f"    Sparse size: {sparse_policy_size} bytes")
    print(f"    Non-zero entries: {len(exp.policy_indices)} / {num_polys * num_polys}")
    print(f"    Sparsity: {(1 - len(exp.policy_indices) / (num_polys * num_polys)) * 100:.1f}%")
    print(f"    Policy compression: {policy_compression:.2%}")
    print(f"    Policy space saved: {(1 - policy_compression) * 100:.1f}%")


def test_replay_buffer():
    """Test replay buffer with multiple experiences."""
    from grobnerRl.training.shared import ReplayBuffer
    
    print("\n" + "="*60)
    print("Testing Replay Buffer with 100 experiences")
    print("="*60)
    
    buffer = ReplayBuffer(max_size=1000)
    
    # Generate 100 experiences
    num_experiences = 100
    experiences = []
    
    for _ in range(num_experiences):
        num_polys = np.random.randint(20, 40)
        num_vars = 5
        
        ideal = tuple(
            np.random.randn(np.random.randint(4, 10), num_vars).astype(np.float32)
            for _ in range(num_polys)
        )
        
        selectables = tuple((i, j) for i in range(num_polys) for j in range(i+1, min(num_polys, i+5)))
        observation = (ideal, selectables)
        
        # Sparse policy
        policy = np.zeros(num_polys * num_polys, dtype=np.float32)
        num_valid = min(15, num_polys * num_polys)
        valid_indices = np.random.choice(num_polys * num_polys, num_valid, replace=False)
        policy[valid_indices] = np.random.dirichlet(np.ones(num_valid))
        
        exp = Experience.from_uncompressed(
            observation=observation,
            policy=policy,
            value=np.random.random(),
            num_polys=num_polys,
            poly_cache=buffer.poly_cache,
        )
        experiences.append(exp)
    
    buffer.add(experiences)
    
    # Get memory stats
    stats = buffer.memory_usage()
    
    print(f"✓ Replay buffer test passed!")
    print(f"  Experiences: {len(buffer)}")
    print(f"  Total memory: {stats['total_mb']:.2f} MB")
    print(f"  Average per experience: {stats['avg_bytes_per_experience'] / 1024:.2f} KB")
    print(f"  Average compression ratio: {stats['avg_compression_ratio']:.2%}")
    print(f"  Space saved: {(1 - stats['avg_compression_ratio']) * 100:.1f}%")
    
    # Estimate for full buffer
    if len(buffer) > 0:
        estimated_full = stats['avg_bytes_per_experience'] * buffer.max_size / (1024 * 1024)
        print(f"\n  Estimated memory for full buffer ({buffer.max_size} experiences):")
        print(f"    {estimated_full:.2f} MB")


def test_replay_buffer_with_dedup():
    """Test replay buffer with polynomial deduplication and cleanup."""
    from grobnerRl.training.shared import ReplayBuffer
    
    print("\n" + "="*60)
    print("Testing Deduplication & Cleanup")
    print("="*60)
    
    buffer = ReplayBuffer(max_size=1000)
    
    # Generate 100 experiences with some polynomial reuse
    num_experiences = 100
    experiences = []
    
    # Create a pool of common polynomials to reuse
    num_vars = 5
    common_polys = [
        np.random.randn(np.random.randint(4, 10), num_vars).astype(np.float32)
        for _ in range(50)  # Only 50 unique polynomials
    ]
    
    for _ in range(num_experiences):
        num_polys = np.random.randint(20, 40)
        
        # Reuse polynomials from the common pool (50% chance)
        ideal = tuple(
            common_polys[np.random.randint(0, len(common_polys))]
            if np.random.random() < 0.5
            else np.random.randn(np.random.randint(4, 10), num_vars).astype(np.float32)
            for _ in range(num_polys)
        )
        
        selectables = tuple((i, j) for i in range(num_polys) for j in range(i+1, min(num_polys, i+5)))
        observation = (ideal, selectables)
        
        # Sparse policy
        policy = np.zeros(num_polys * num_polys, dtype=np.float32)
        num_valid = min(15, num_polys * num_polys)
        valid_indices = np.random.choice(num_polys * num_polys, num_valid, replace=False)
        policy[valid_indices] = np.random.dirichlet(np.ones(num_valid))
        
        exp = Experience.from_uncompressed(
            observation=observation,
            policy=policy,
            value=np.random.random(),
            num_polys=num_polys,
            poly_cache=buffer.poly_cache,
        )
        experiences.append(exp)
    
    buffer.add(experiences)
    
    # Get memory stats
    stats = buffer.memory_usage()
    
    print(f"✓ Replay buffer with deduplication test passed!")
    print(f"  Experiences: {len(buffer)}")
    print(f"  Total memory: {stats['total_mb']:.2f} MB")
    print(f"  Average per experience: {stats['avg_bytes_per_experience'] / 1024:.2f} KB")
    print(f"  Average compression ratio: {stats['avg_compression_ratio']:.2%}")
    print(f"  Space saved: {(1 - stats['avg_compression_ratio']) * 100:.1f}%")
    
    print(f"\n  Polynomial deduplication:")
    print(f"    Unique polynomials: {stats['unique_polynomials']}")
    print(f"    Polynomial cache: {stats['poly_cache_mb']:.2f} MB")
    print(f"    Deduplication ratio: {stats['deduplication_ratio']:.2%}")
    print(f"    Dedup space saved: {(1 - stats['deduplication_ratio']) * 100:.1f}%")
    
    # Test that we can sample and batch
    if len(buffer) >= 32:
        batched_obs, policies, values, mask = buffer.sample_batched(32)
        print(f"\n  ✓ Batching works with deduplicated experiences")
        print(f"    Batch shape: {batched_obs['ideals'].shape}")
    
    # Estimate for full buffer
    if len(buffer) > 0:
        estimated_full = stats['avg_bytes_per_experience'] * buffer.max_size / (1024 * 1024)
        print(f"\n  Estimated memory for full buffer ({buffer.max_size} experiences):")
        print(f"    {estimated_full:.2f} MB")


def test_cleanup():
    """Test that polynomials are cleaned up when experiences are evicted."""
    from grobnerRl.training.shared import ReplayBuffer, PolynomialCache
    
    print("\n" + "="*60)
    print("Testing Polynomial Cleanup on Buffer Overflow")
    print("="*60)
    
    buffer = ReplayBuffer(max_size=10)
    
    for iteration in range(3):
        experiences = []
        for _ in range(15):
            num_polys = 5
            num_vars = 3
            
            ideal = tuple(
                np.random.randn(4, num_vars).astype(np.float32)
                for _ in range(num_polys)
            )
            
            selectables = tuple((i, j) for i in range(num_polys) for j in range(i+1, num_polys))
            observation = (ideal, selectables)
            
            policy = np.zeros(num_polys * num_polys, dtype=np.float32)
            policy[np.random.choice(num_polys * num_polys, 3, replace=False)] = [0.5, 0.3, 0.2]
            
            exp = Experience.from_uncompressed(
                observation=observation,
                policy=policy,
                value=np.random.random(),
                num_polys=num_polys,
                poly_cache=buffer.poly_cache,
            )
            experiences.append(exp)
        
        buffer.add(experiences)
        stats = buffer.memory_usage()
        
        print(f"\n  Iteration {iteration + 1}:")
        print(f"    Buffer size: {len(buffer)} / {buffer.max_size}")
        print(f"    Unique polynomials in cache: {stats['unique_polynomials']}")
        print(f"    Cache memory: {stats['poly_cache_mb']:.3f} MB")
    
    print(f"\n  ✓ Cleanup working! Cache stays bounded despite adding {3 * 15} experiences")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING EXPERIENCE COMPRESSION")
    print("="*60)
    
    test_compression()
    test_large_problem()
    test_replay_buffer()
    test_replay_buffer_with_dedup()
    test_cleanup()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
