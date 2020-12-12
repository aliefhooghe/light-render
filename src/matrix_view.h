#ifndef XREDNER_MATRIX_VIEW_H_
#define XREDNER_MATRIX_VIEW_H_

namespace Xrender  {

    /**
     * \brief matrix_view  provide a 2D view on a 1D container
     * \note element are supposed to be stored by line, starting from the bottom
     *  
     *  y
     *  ^
     *  |
     *  |
     *  |________> x 
     */
    template <typename Tarray>
    class matrix_view {

    public:
        matrix_view(Tarray& array, std::size_t width, std::size_t height)
        :   _ref{array}, _width{width}
        {}

        auto& operator() (const std::size_t x, const std::size_t y) noexcept
        {
            return _ref[x + _width * y];
        }

        const auto& operator() (const std::size_t x, const std::size_t y) const noexcept
        {
            return _ref[x + _width * y];
        }

    private:
        Tarray& _ref;
        const std::size_t _width;
    };

}

#endif  /* XREDNER_MATRIX_VIEW_H_ */