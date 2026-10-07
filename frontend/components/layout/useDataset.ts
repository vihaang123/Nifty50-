"use client";
import { getDataset } from "@/lib/api";
import { useRequest } from "@/lib/hooks";

export const useDataset = () => useRequest(getDataset, "dataset");
