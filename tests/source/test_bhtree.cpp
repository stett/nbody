#include <random>
#include <catch2/catch_test_macros.hpp>
#include "nbody/bhtree.h"
#include "nbody/vector.h"

using nbody::Body;
using nbody::Vector;
using nbody::bh::Node;
using nbody::bh::Tree;

namespace
{
    bool compare(const Vector& a, const Vector& b) {
        return
            std::abs(a.x - b.x) < std::numeric_limits<float>::epsilon() &&
            std::abs(a.y - b.y) < std::numeric_limits<float>::epsilon() &&
            std::abs(a.z - b.z) < std::numeric_limits<float>::epsilon();
    }
}

TEST_CASE("create tree", "[bh tree]")
{
    Tree tree({ .size=100 });
}

TEST_CASE("insert 2 particles in different quadrants", "[bh tree 3]")
{
	Tree tree({ .size=100 });
	std::vector<Body> bodies;

	const float m0 = 1;
	const Vector p0 = { 1,1,1 };
	const float m1 = 2;
	const Vector p1 = { -1,-1,-1 };

    // after inserting first
	bodies.push_back({ .mass=m0, .pos=p0 });
	tree.build(bodies);
    //REQUIRE(tree.nodes().size() == 1);
    REQUIRE(tree.nodes()[0].mass == m0);
    REQUIRE(tree.nodes()[0].com == p0);

    // after inserting second
    bodies.push_back({ .mass=m1, .pos=p1 });
    tree.build(bodies);
	//REQUIRE(tree.nodes().size() == 1+8);
	REQUIRE(tree.nodes()[0].mass == m0 + m1);
	REQUIRE(compare(tree.nodes()[0].com, ((m0 * p0) + (m1 * p1)) / (m0 + m1)));

	// test leaf nodes
	const uint8_t q0 = tree.bounds().quadrant(p0);
	const uint8_t q1 = tree.bounds().quadrant(p1);
	const int32_t c0 = tree.nodes()[0].children + q0;
    const int32_t c1 = tree.nodes()[0].children + q1;
    REQUIRE(tree.nodes()[c0].mass == m0);
    REQUIRE(compare(tree.nodes()[c0].com, p0));
    // NOTE: ... just realized, this isn't going to work with the parallel insertion because
    // in that scheme children can't be next to each other :|
    REQUIRE(tree.nodes()[c1].mass == m1);
    REQUIRE(compare(tree.nodes()[c1].com, p1));
}

/*
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

TEST_CASE("insert 2 particles in the same location", "[bh tree 3]")
{
	Tree tree({ .size = 100 });
	tree.insert({1,1,1}, 1);
	tree.insert({1,1,1}, 1);
}

TEST_CASE("insert 2 particles very close to each other", "[bh tree 3]")
{
	Tree tree({ .size = 100 });
	tree.insert({1,1,1}, 1);
	tree.insert({1 + std::numeric_limits<float>::epsilon(),1,1}, 1);
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
}

TEST_CASE("insert 1000000 particles", "[bh tree 3]")
{
	const float size = 100;
	Tree tree({ .size = size * 2 });
	const size_t num = 1000000; // million
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
}

TEST_CASE("apply with 1 far away particle", "[bh tree 3]")
{
	Tree tree({ .size=200 });
	const Vector p0{ 100,100,100 };
	const float m0 = 1;
	tree.insert(p0, m0);
	tree.apply({ 0,0,0 }, [&](const Node& node)//const Vector& pos, float mass)
	{
		REQUIRE(node.com == p0);
		REQUIRE(node.mass == m0);
	});
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
	tree.apply({ 0,0,0 }, [&](const Node& node)//const Vector& pos, float mass)
	{
		REQUIRE(node.com == ((p0*m0)+(p1*m1))/(m0+m1));
		REQUIRE(node.mass == m0+m1);
	});
}

TEST_CASE("compare n^2 gravitation to n*log(n) gravitation with 100 particles", "[bh tree 3]")
{
	// setup test parameters
	static constexpr float G = 1.f;
	static constexpr float epsilon = .5f;
	static constexpr float precision = 1000 * std::numeric_limits<float>::epsilon();

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

		// compute force of gravity
		const float f_mag = (G * mass0 * mass1) / delta_sq;
		return f_mag * f_mag * delta * delta / delta_sq;
	};

	size_t interactions_n2 = 0;
	size_t interactions_nlogn = 0;
	for (size_t i = 0; i < num; ++i)
	{
		// compute n2 force
		Vector force_n2 = { 0,0,0 };
		for (size_t j = 0; j < num; ++j)
		{
			force_n2 += compute_force(positions[i], 1, positions[j], 1);
			++interactions_n2;
		}

		// compute nlogn force
		Vector force_nlogn = { 0,0,0 };
		tree.apply(positions[i], [&](const Node& node) {//const Vector& pos, float) {
			force_nlogn += compute_force(positions[i], 1, node.com, 1);
			++interactions_nlogn;
		}, epsilon);

		// compare the forces
		const float force_diff_sq = force_nlogn.dist_sq(force_n2);
		REQUIRE(force_diff_sq < precision);
	}

	REQUIRE(0 < interactions_nlogn);
	REQUIRE(interactions_nlogn < interactions_n2);
}
*/
