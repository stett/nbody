#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "nbody/bhtree.h"
#include "nbody/vector.h"

using nbody::Vector;
using nbody::bh::Node;
using nbody::bh::Tree;

namespace
{
    // Relative comparison. An absolute FLT_EPSILON bound would be stricter than exact
    // bit equality is forgiving at these magnitudes -- 1 ULP at 50.0 is already ~3.8e-6,
    // some 30x FLT_EPSILON -- so it passed only because the tree happened to compute the
    // expression in the same order the test does. Any harmless reassociation inside
    // Tree::accumulate would have broken it.
    bool compare(const Vector& a, const Vector& b) {
        const auto close = [](const float x, const float y) {
            const float scale = std::max({ 1.f, std::abs(x), std::abs(y) });
            return std::abs(x - y) <= 1e-5f * scale;
        };
        return close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z);
    }
}

TEST_CASE("create tree", "[bh tree 3]")
{
    Tree tree({ .size=100 });

    // A fresh tree holds its root inline and nothing else.
    REQUIRE(tree.nodes().size() == 1);
    REQUIRE(tree.nodes()[0].mass == 0.f);
    REQUIRE(tree.bounds().size == 100.f);
}

TEST_CASE("insert 2 particles in different quadrants", "[bh tree 3]")
{
	Tree tree({ .size=100 });

	const float m0 = 1;
	const Vector p0 = { 1,1,1 };
	const float m1 = 2;
	const Vector p1 = { -1,-1,-1 };

    // after inserting first
	tree.insert(p0, m0);
    REQUIRE(tree.nodes().size() == 1);
    REQUIRE(tree.nodes()[0].mass == m0);
    REQUIRE(tree.nodes()[0].com == p0);

    // after inserting second
    tree.insert(p1, m1);
	REQUIRE(tree.nodes().size() == 1+8);
	REQUIRE(tree.nodes()[0].mass == m0 + m1);
	REQUIRE(compare(tree.nodes()[0].com, ((m0 * p0) + (m1 * p1)) / (m0 + m1)));

	// test leaf nodes
	const uint8_t q0 = tree.bounds().quadrant(p0);
	const uint8_t q1 = tree.bounds().quadrant(p1);
	REQUIRE(tree.nodes()[1 + q0].mass == m0);
	REQUIRE(compare(tree.nodes()[1 + q0].com, p0));
	REQUIRE(tree.nodes()[1 + q1].mass == m1);
	REQUIRE(compare(tree.nodes()[1 + q1].com, p1));
}

TEST_CASE("insert 2 particles in the same quadrant", "[bh tree 3]")
{
	Tree tree({ .size = 100 });

	const float m0 = 1;
	const Vector p0 = { 1,1,1 };
	const float m1 = 1;
	const Vector p1 = { 99,99,99 };

    // After inserting first
	tree.insert(p0, m0);
    REQUIRE(tree.nodes().size() == 1);
    REQUIRE(tree.nodes()[0].mass == m0);
    REQUIRE(tree.nodes()[0].com == p0);

    // After inserting second
    tree.insert(p1, m1);
	REQUIRE(tree.nodes().size() == 1+8+8);
	REQUIRE(tree.nodes()[0].mass == m0 + m1);
	REQUIRE(compare(tree.nodes()[0].com, ((m0 * p0) + (m1 * p1)) / (m0 + m1)));

	// Intermediate node
	uint8_t q0 = tree.bounds().quadrant(p0);
	uint8_t q1 = tree.bounds().quadrant(p1);
	uint32_t c = tree.nodes()[0].children;
	const uint32_t i = c + q0;
	REQUIRE(q0 == q1);
	REQUIRE(tree.nodes()[i].mass == m0 + m1);
	REQUIRE(compare(tree.nodes()[i].com, ((m0 * p0) + (m1 * p1)) / (m0 + m1)));

	// Leaf nodes
	q0 = tree.nodes()[i].bounds.quadrant(p0);
	q1 = tree.nodes()[i].bounds.quadrant(p1);
	c = tree.nodes()[i].children;
	REQUIRE(q0 != q1);
	REQUIRE(tree.nodes()[c + q0].mass == m0);
	REQUIRE(compare(tree.nodes()[c + q0].com, p0));
	REQUIRE(tree.nodes()[c + q1].mass == m1);
	REQUIRE(compare(tree.nodes()[c + q1].com, p1));
}

TEST_CASE("insert 8 particles in different quadrants", "[bh tree 3]")
{
    Tree tree({ .size = 100 });

    // q0
    tree.insert({50,50,50}, 1);
    REQUIRE(tree.nodes().size() == 1);
    REQUIRE(tree.nodes()[0].mass == 1);
    REQUIRE(tree.nodes()[0].com == Vector{50,50,50});

    // q1
    tree.insert({-50,50,50}, 1);
    REQUIRE(tree.nodes().size() == 9);
    REQUIRE(tree.nodes()[0].mass == 2);
    REQUIRE(tree.nodes()[0].com == Vector{0,50,50});
    REQUIRE(tree.nodes()[1].mass == 1);
    REQUIRE(tree.nodes()[1].com == Vector{50,50,50});
    REQUIRE(tree.nodes()[2].mass == 1);
    REQUIRE(tree.nodes()[2].com == Vector{-50,50,50});

    // q2
    tree.insert({50,-50,50}, 1);
    REQUIRE(tree.nodes().size() == 9);
    REQUIRE(tree.nodes()[0].mass == 3);
    //REQUIRE(tree.nodes[0].com == Vector{??,??,??})
    REQUIRE(tree.nodes()[1].mass == 1);
    REQUIRE(tree.nodes()[1].com == Vector{50,50,50});
    REQUIRE(tree.nodes()[2].mass == 1);
    REQUIRE(tree.nodes()[2].com == Vector{-50,50,50});
    REQUIRE(tree.nodes()[3].mass == 1);
    REQUIRE(tree.nodes()[3].com == Vector{50,-50,50});
}

// Coincident and near-coincident bodies drive subdivision until the bounds shrink below
// the cutoff in Tree::insert. These would otherwise recurse forever, so the point is
// that they terminate -- but assert the outcome too, or a regression that subdivided a
// level too few would still "pass" by merely not hanging.
TEST_CASE("insert 2 particles in the same location", "[bh tree 3]")
{
	Tree tree({ .size = 100 });
	tree.insert({1,1,1}, 1);
	tree.insert({1,1,1}, 1);

	REQUIRE(tree.nodes().size() > 1);
	REQUIRE(tree.nodes()[0].mass == 2.f);
	REQUIRE(compare(tree.nodes()[0].com, Vector{1,1,1}));
}

TEST_CASE("insert 2 particles very close to each other", "[bh tree 3]")
{
	Tree tree({ .size = 100 });
	tree.insert({1,1,1}, 1);
	tree.insert({1 + std::numeric_limits<float>::epsilon(),1,1}, 1);

	REQUIRE(tree.nodes().size() > 1);
	REQUIRE(tree.nodes()[0].mass == 2.f);
}

TEST_CASE("insert 100 particles", "[bh tree 3]")
{
	const float size = 100;
	Tree tree({ .size=size*2 });
	const size_t num = 100;
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0, .5f);
	const size_t num_nodes = num * num;
	tree.reserve(num_nodes);
	for (size_t i = 0; i < num; ++i)
	{
		tree.insert({
			.x = std::clamp(distribution(generator), -1.f, 1.f) * size,
			.y = std::clamp(distribution(generator), -1.f, 1.f) * size,
			.z = std::clamp(distribution(generator), -1.f, 1.f) * size
		}, 1.f);
	}

	// total mass is conserved and every body landed somewhere in the tree
	REQUIRE(tree.nodes().size() > 1);
	REQUIRE(tree.nodes()[0].mass == float(num));
}

TEST_CASE("insert 100000 particles", "[bh tree 3]")
{
	const float size = 100;
	Tree tree({ .size = size * 2 });
	const size_t num = 100000; // hundred thousand
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0, .5f);
	const size_t num_nodes = 10 * num;
	tree.reserve(num_nodes);
	for (size_t i = 0; i < num; ++i)
	{
		tree.insert({
			.x = std::clamp(distribution(generator), -1.f, 1.f) * size,
			.y = std::clamp(distribution(generator), -1.f, 1.f) * size,
			.z = std::clamp(distribution(generator), -1.f, 1.f) * size
			}, 1.f);
	}

	REQUIRE(tree.nodes().size() > 1);
	REQUIRE(tree.nodes()[0].mass == float(num));
}

// NOTE: there was a 1,000,000-particle case here too. It exercised no code path this
// one does not, but reserve(10 * num) asks for a single contiguous 480 MB block, which
// is a hard failure on a memory-capped CI container rather than a useful signal.

TEST_CASE("apply with 1 far away particle", "[bh tree 3]")
{
	Tree tree({ .size=200 });
	const Vector p0{ 100,100,100 };
	const float m0 = 1;
	tree.insert(p0, m0);

	// Count the visits: assertions living only inside the callback would all vanish,
	// and the test still pass, if apply() regressed to never invoking the visitor.
	size_t visits = 0;
	tree.apply({ 0,0,0 }, [&](const Node& node)
	{
		++visits;
		REQUIRE(compare(node.com, p0));
		REQUIRE(node.mass == m0);
	});
	REQUIRE(visits == 1);
}

TEST_CASE("apply with 2 far away particles", "[bh tree 3]")
{
	Tree tree({ .size=200 });
	const Vector p0{ 100,100,100 };
	const float m0 = 1;
	const Vector p1{ 99,99,99 };
	const float m1 = 1;
	tree.insert(p0, m0);
	tree.insert(p1, m1);

	size_t visits = 0;
	tree.apply({ 0,0,0 }, [&](const Node& node)
	{
		++visits;
		REQUIRE(compare(node.com, ((p0*m0)+(p1*m1))/(m0+m1)));
		REQUIRE(node.mass == m0+m1);
	});

	// both bodies are far enough away to collapse into a single node
	REQUIRE(visits == 1);
}

TEST_CASE("compare n^2 gravitation to n*log(n) gravitation with 100 particles", "[bh tree 3]")
{
	// setup test parameters
	static constexpr float G = 1.f;
	static constexpr float epsilon = .5f;
	// Errors are judged against the mean force magnitude rather than per-body. A
	// per-body relative bound is the wrong shape here: with 100 equal-mass bodies in a
	// random cloud the net force nearly cancels for some of them, so |f| is tiny and the
	// relative error exceeds 100% while the absolute error stays small.

	// create tree
	const float size = 100;
	Tree tree({ .size = size * 2 });
	const size_t num = 100;
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0, .5);
	std::vector<Vector> positions;
	positions.reserve(num);
	for (size_t i = 0; i < num; ++i)
	{
		positions.push_back({
			.x = std::clamp(distribution(generator), -1.f, 1.f) * size,
			.y = std::clamp(distribution(generator), -1.f, 1.f) * size,
			.z = std::clamp(distribution(generator), -1.f, 1.f) * size
		});
		tree.insert(positions.back(), 1.f);
	}

	// function for computing force
	const auto compute_force = [](const Vector& pos0, float mass0, const Vector& pos1, float mass1) -> Vector
	{
		// if we're too close, don't apply a force
		Vector delta = pos1 - pos0;
		float delta_sq = dot(delta, delta);
		if (delta_sq < std::numeric_limits<float>::epsilon())
			return { 0,0,0 };

		// Compute force of gravity: magnitude G*m0*m1/r^2 along the unit separation.
		// This previously read `f_mag * f_mag * delta * delta / delta_sq`, which
		// squares an already-complete magnitude and multiplies by the component-wise
		// square of delta -- not gravity, and always pointing into the positive octant.
		// It was self-consistent across both sides so the comparison still held, but it
		// was not comparing the thing the test claims to compare.
		const float f_mag = (G * mass0 * mass1) / delta_sq;
		return f_mag * delta / std::sqrt(delta_sq);
	};

	size_t interactions_n2 = 0;

	// exact forces, and the scale to judge errors against
	std::vector<Vector> exact(num);
	double total_force_magnitude = 0;
	for (size_t i = 0; i < num; ++i)
	{
		for (size_t j = 0; j < num; ++j)
		{
			exact[i] += compute_force(positions[i], 1, positions[j], 1);
			++interactions_n2;
		}
		total_force_magnitude += std::sqrt(exact[i].size_sq());
	}
	const double mean_force = total_force_magnitude / double(num);
	REQUIRE(mean_force > 0.0);

	// Run the tree at a given opening parameter and report the worst error, scaled by
	// the mean force, plus how much work it did.
	struct Result { double error; size_t interactions; };
	const auto run = [&](const float theta) -> Result
	{
		double worst = 0;
		size_t interactions = 0;
		for (size_t i = 0; i < num; ++i)
		{
			Vector approx = { 0,0,0 };
			tree.apply(positions[i], [&](const Node& node) {
				approx += compute_force(positions[i], 1, node.com, 1);
				++interactions;
			}, theta);
			worst = std::max<double>(worst, std::sqrt(approx.dist_sq(exact[i])));
		}
		return { worst / mean_force, interactions };
	};

	// NOTE ON THETA: this tree accepts a node when `dist > node_size * theta`
	// (source/bhtree.cpp:118). Textbook Barnes-Hut accepts when `dist > node_size /
	// theta`, so theta here is INVERTED relative to convention: larger is more accurate,
	// not less. At the library default of 0.5 a node is accepted from only half its own
	// width away, which is a very aggressive approximation.
	const Result loose = run(epsilon);   // 0.5, the library default
	const Result tight = run(8.f);       // ~ conventional theta of 0.125

	INFO("theta 0.5 -> error " << loose.error << " over " << loose.interactions << " interactions; "
		<< "theta 8 -> error " << tight.error << " over " << tight.interactions << " interactions");

	// Opening the criterion up must converge on the exact answer...
	REQUIRE(tight.error < loose.error);
	REQUIRE(tight.error < .05);

	// ...at the cost of more work, but still less than the exhaustive sum. That second
	// half matters: without it the test would pass just as happily if apply() had
	// regressed to visiting every body individually.
	REQUIRE(tight.interactions > loose.interactions);
	REQUIRE(0 < loose.interactions);
	REQUIRE(tight.interactions < interactions_n2);

	// Pins the default's accuracy as it actually is, rather than as one might assume.
	// If the theta convention is ever corrected this should get dramatically tighter.
	REQUIRE(loose.error < 3.0);
}
