#include <gtest/gtest.h>

#include "ERF_SLM_NLCD40.H"

TEST(SLMNLCD40, DistinctUrbanAndForestRoughness)
{
    const auto z0 = [](int category, amrex::Real fraction,
                       amrex::Real minimum, amrex::Real maximum) {
        const auto& row = SLMNLCD40::values[category - 1];
        const amrex::Real weight = SLMNLCD40::fraction_weight(fraction, minimum, maximum);
        return row[SLMNLCD40::Z0Min] +
               weight * (row[SLMNLCD40::Z0Max] - row[SLMNLCD40::Z0Min]);
    };

    EXPECT_DOUBLE_EQ(z0(23, 0.5, 0.1, 0.8), 0.5);
    EXPECT_DOUBLE_EQ(z0(24, 0.5, 0.1, 0.8), 0.7);
    EXPECT_DOUBLE_EQ(z0(25, 0.5, 0.1, 0.8), 1.5);
    EXPECT_DOUBLE_EQ(z0(26, 0.5, 0.1, 0.8), 2.0);
    EXPECT_DOUBLE_EQ(z0(28, 0.5, 0.1, 0.8), 0.5);
    EXPECT_DOUBLE_EQ(z0(29, 0.5, 0.1, 0.8), 0.5);
    EXPECT_DOUBLE_EQ(z0(30, 0.1, 0.1, 0.8), 0.2);
    EXPECT_DOUBLE_EQ(z0(30, 0.8, 0.1, 0.8), 0.5);
    EXPECT_NEAR(z0(30, 0.5, 0.5, 0.5), 0.35, 1.e-12);
    EXPECT_DOUBLE_EQ(z0(30, 1.0, 0.1, 0.8), 0.5);
}

TEST(SLMNLCD40, MODISOpticsAnalogsRemainInRange)
{
    for (int category = 22; category <= 40; ++category) {
        EXPECT_GE(SLMNLCD40::analog(category), 1);
        EXPECT_LE(SLMNLCD40::analog(category), 16);
    }
    EXPECT_EQ(SLMNLCD40::analog(22), 15);
    EXPECT_EQ(SLMNLCD40::analog(23), 13);
    EXPECT_EQ(SLMNLCD40::analog(26), 13);
    EXPECT_EQ(SLMNLCD40::analog(28), 4);
    EXPECT_EQ(SLMNLCD40::analog(29), 1);
    EXPECT_EQ(SLMNLCD40::analog(40), 11);
}

TEST(SLMNLCD40, UndefinedClassesAreRejected)
{
    EXPECT_TRUE(SLMNLCD40::valid_category(0));
    EXPECT_TRUE(SLMNLCD40::valid_category(17));
    EXPECT_TRUE(SLMNLCD40::valid_category(21));
    EXPECT_TRUE(SLMNLCD40::valid_category(40));
    for (int category : {18, 19, 20, 41, -1}) {
        EXPECT_FALSE(SLMNLCD40::valid_category(category));
    }
}
