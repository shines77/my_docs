#ifndef _LEAVES_SIGNAL_SLOT_HPP_
#define _LEAVES_SIGNAL_SLOT_HPP_

#include <functional>
#include <map>
#include <array>

namespace leaves
{
	namespace detail
	{
		template <typename T>
		struct BKDRHash
		{
			static_assert(std::is_integral<T>::value, "只处理整型数据!");

			typedef T DataType;

			template <typename Iterator>
			DataType operator() (Iterator begin, Iterator end)
			{
				DataType seed = 131;
				DataType hash = 0;

				Iterator itr = begin;
				while (itr != end)
				{
					hash = hash * seed + (*itr++);
				}

				return hash % 0x7FFFFFFF;
			}
		};

		template 
		<
			typename T,
			template <typename, std::size_t> class ArrayPolicy = std::array,
			template <typename> class HashPolicy = BKDRHash
		>
		struct SignatureBit
		{
			enum { ArraySize = 2 };

			typedef T DataType;
			typedef ArrayPolicy<DataType, ArraySize> ArrayType;
			typedef HashPolicy<DataType> HashFunctionType;

			template <typename ... Args>
			explicit SignatureBit(void(*slot)(Args...))
			{
				typedef void(*PToFunc)(Args...);

				ArrayType tempData;
				*reinterpret_cast<PToFunc*>(&tempData[0]) = slot;

				Data_ = HashFunctionType()(tempData.begin(), tempData.end());
			}

			template <typename R, typename ... Args>
			explicit SignatureBit(R* r, void(R::*slot)(Args...))
			{
				typedef void(R::*PToMFunc)(Args...);

				ArrayType tempData;
				tempData[0] = (DataType)r;
				*reinterpret_cast<PToMFunc*>(&tempData[1]) = slot;

				Data_ = HashFunctionType()(tempData.begin(), tempData.end());
			}

			operator DataType() const
			{
				return Data_;
			}

			DataType	Data_;
		};

		class Connection
		{
		public:
			template <typename Signal, typename R, typename ... Args>
			Connection(Signal& signal, R* r, void(R::*slot)(Args...))
			{
				signal.connectInternal(r, slot);
			}

			template <typename Signal, typename ... Args>
			Connection(Signal& signal, void(*slot)(Args...))
			{
				signal.connectInternal(slot);
			}
		};

		class Disconnection
		{
		public:
			template <typename Signal, typename R, typename ... Args>
			Disconnection(Signal& signal, R* r, void(R::*slot)(Args...))
			{
				signal.disconnectInternal(r, slot);
			}

			template <typename Signal, typename ... Args>
			Disconnection(Signal& signal, void(*slot)(Args...))
			{
				signal.disconnectInternal(slot);
			}
		};

		class Emit
		{
		public:
			template <typename Signal, typename ... Args>
			Emit(Signal& signal, Args&& ... args)
			{
				signal(std::forward<Args>(args)...);
			}
		};
	}

	template <typename F>
	class Signal;

	template <typename ... Args>
	class Signal<void(Args...)>
	{
		friend class detail::Connection;
		friend class detail::Disconnection;
		friend class detail::Emit;

		typedef detail::SignatureBit<std::size_t> SignatureType;
		typedef std::function<void(Args...)> FunctionType;
		typedef std::map<SignatureType, FunctionType> SlotContainer;

		template <typename ... ArgsT>
		inline void connectInternal(void(*slot)(ArgsT...))
		{
			Slots_.insert(std::make_pair(SignatureType{ slot }, FunctionType{ slot }));
		}

		template <typename T, typename ... ArgsT>
		inline void connectInternal(T* t, void(T::*slot)(ArgsT...))
		{
			FunctionType function = [=](ArgsT ... args){ (t->*slot)(args...); };
			Slots_.insert(std::make_pair(SignatureType{ t, slot }, std::move(function)));
		}

		template <typename ... ArgsT>
		inline void disconnectInternal(void(*slot)(ArgsT...))
		{
			Slots_.erase(SignatureType{ slot });
		}

		template <typename T, typename ... ArgsT>
		inline void disconnectInternal(T* t, void(T::*slot)(ArgsT...))
		{
			Slots_.erase(SignatureType{ t, slot });
		}

		inline void operator() (Args&&... args)
		{
			for (auto& elem : Slots_)
			{
				elem.second(std::forward<Args>(args)...);
			}
		}

		SlotContainer Slots_;
	};

	template <typename EmitterT, typename Emitter, typename ReceiverT, typename Receiver, typename ... Args>
	inline void connect(EmitterT* emitter, Signal<void(Args...)> Emitter::*signal, ReceiverT* receiver, void(Receiver::*slot)(Args...))
	{
		static_assert(std::is_base_of<Emitter, EmitterT>::value, "Emitter Type illegal!");
		static_assert(std::is_base_of<Receiver, ReceiverT>::value, "Receiver Type illegal!");
		detail::Connection(dynamic_cast<Emitter*>(emitter)->*signal, receiver, slot);
	}

	template <typename EmitterT, typename Emitter, typename ... Args>
	inline void connect(EmitterT* emitter, Signal<void(Args...)>Emitter::*signal, void(*slot)(Args...))
	{
		static_assert(std::is_base_of<Emitter, EmitterT>::value, "Type illegal!");
		detail::Connection(dynamic_cast<Emitter*>(emitter)->*signal, slot);
	}

	template <typename EmitterT, typename Emitter, typename ReceiverT, typename Receiver, typename ... Args>
	inline void disconnect(EmitterT* emitter, Signal<void(Args...)> Emitter::*signal, ReceiverT* receiver, void(Receiver::*slot)(Args...))
	{
		static_assert(std::is_base_of<Emitter, EmitterT>::value, "Emitter Type illegal!");
		static_assert(std::is_base_of<Receiver, ReceiverT>::value, "Receiver Type illegal!");
		detail::Disconnection(dynamic_cast<Emitter*>(emitter)->*signal, receiver, slot);
	}

	template <typename EmitterT, typename Emitter, typename ... Args>
	inline void disconnect(EmitterT* emitter, Signal<void(Args...)>Emitter::*signal, void(*slot)(Args...))
	{
		static_assert(std::is_base_of<Emitter, EmitterT>::value, "Type illegal!");
		detail::Disconnection(dynamic_cast<Emitter*>(emitter)->*signal, slot);
	}

	template <typename Signal, typename ... Args>
	inline void emit(Signal& signal, Args&& ... args)
	{
		detail::Emit(signal, std::forward<Args>(args)...);
	}

}

#endif